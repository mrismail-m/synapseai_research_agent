"""
Research Agent Module.

Handles web search via Tavily and interprets the search results using Groq.
Refines user queries based on context and produces structured findings.
"""

import json
import re
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import AnyMessage
from src.config import config
from src.graph.state import AgentState
from src.tools.search import get_tavily_tool

class ResearchAnalysis(BaseModel):
    """Analysis of research findings."""
    findings: str = Field(description="Comprehensive summary of news, financials, leadership, and developments.")
    confidence_score: int = Field(description="Confidence score from 0-10 based on data quality and specificity.", ge=0, le=10)
    summary_message: str = Field(description="A brief summary message for the user.")

def _build_refiner_prompt(context: List[AnyMessage], query: str) -> str:
    """Helper to build the query refinement prompt."""
    return (
        "Based on the conversation history below, what is the specific company and topic we are researching? "
        "Provide ONLY the search query text, no other conversation.\n\n"
        f"Context:\n{context}\n\n"
        f"Latest User Query: {query}\n\n"
        "Search Query:"
    )

def _build_analysis_prompt(search_results: str) -> str:
    """Helper to build the research analysis prompt."""
    return (
        "You are a Senior Research Analyst. Analyze the search results and return a JSON object with exactly these keys:\n"
        "1. 'findings': string summary of news, financials, leadership.\n"
        "2. 'confidence_score': integer 0-10.\n"
        "3. 'summary_message': short string for the user.\n\n"
        "Do not include any other text or markdown tags outside the JSON.\n\n"
        f"Search Results:\n{search_results}"
    )

def research_agent(state: AgentState) -> Dict[str, Any]:
    """
    Researches the company using Tavily and analyzes findings with Groq.
    Uses conversation history to maintain context.
    
    Args:
        state (AgentState): The current graph state.
        
    Returns:
        dict: The state updates including research findings, confidence score, and attempts.
    """
    search_tool = get_tavily_tool()
    
    messages = state.get("messages", [])
    query = state.get("query", "")
    
    refiner_llm = ChatGroq(
        model=config.LLM_MODEL_NAME, 
        api_key=config.GROQ_API_KEY, 
        temperature=config.LLM_TEMPERATURE_STRICT
    )
    
    context = messages[-4:] if len(messages) >= 4 else messages
    refiner_prompt = _build_refiner_prompt(context, query)
    
    # 1. Refine and Execute Search
    try:
        search_query = refiner_llm.invoke(refiner_prompt).content.strip().replace('"', '')
        search_results = search_tool.invoke(search_query)
    except Exception as e:
        # Graceful fallback if search fails
        search_results = f"Search failed: {str(e)}"
        return {
            "research_findings": f"Could not complete research. Error: {str(e)}",
            "confidence_score": 0,
            "attempts": state.get("attempts", 0) + 1
        }
    
    # 2. Analyze with LLM
    llm = ChatGroq(
        model=config.LLM_MODEL_NAME,
        api_key=config.GROQ_API_KEY,
        temperature=config.LLM_TEMPERATURE_CREATIVE
    )
    
    system_prompt = _build_analysis_prompt(str(search_results))
    
    try:
        # Try structured output first
        structured_llm = llm.with_structured_output(ResearchAnalysis)
        analysis = structured_llm.invoke(system_prompt)
    except Exception:
        # Fallback to raw parsing or handle complete LLM failure
        try:
            raw_res = llm.invoke(system_prompt).content
            json_str = re.search(r'\{.*\}', raw_res, re.DOTALL).group()
            data = json.loads(json_str)
            analysis = ResearchAnalysis(**data)
        except Exception as e:
            analysis = ResearchAnalysis(
                findings=f"Analysis failed. Raw search data: {str(search_results)[:500]}...",
                confidence_score=0,
                summary_message=f"Analysis encountered an error: {str(e)}"
            )
    
    attempts = state.get("attempts", 0) + 1
    
    return {
        "research_findings": analysis.findings,
        "confidence_score": analysis.confidence_score,
        "attempts": attempts
    }
