"""
Validator Agent Module.

Responsible for checking the output of the research agent against the user's
original query to ensure the response is sufficiently detailed and accurate.
"""

from typing import Literal, Dict, Any
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from src.config import config
from src.graph.state import AgentState

class ValidationResult(BaseModel):
    """Validation of research findings sufficiency."""
    status: Literal["sufficient", "insufficient"] = Field(description="Whether the findings answer the query.")
    feedback: str = Field(description="Feedback on what is missing if insufficient.")

def validator_agent(state: AgentState) -> Dict[str, Any]:
    """
    Strictly validates if research findings adequately address the specific user query.
    
    Args:
        state (AgentState): The current graph state.
        
    Returns:
        dict: The state updates containing the validation result.
    """
    llm = ChatGroq(
        model=config.LLM_MODEL_NAME,
        api_key=config.GROQ_API_KEY,
        temperature=config.LLM_TEMPERATURE_STRICT
    )
    
    structured_llm = llm.with_structured_output(ValidationResult)
    
    findings = state.get("research_findings", "")
    query = state.get("query", "")
    
    system_prompt = (
        "You are a Quality Control Auditor for a research team. Your job is to decide if the "
        "current research findings adequately answer the user's SPECIFIC question.\n\n"
        "STRICT VALIDATION RULES:\n"
        "1. If the user asked about 'competitors', the findings MUST list real, specific "
        "competitors and briefly explain their relationship to the subject.\n"
        "2. If findings are generic, vague, or do not directly address the query, mark 'insufficient'.\n"
        "3. High specificity is required. Don't accept 'No data found' if you believe data exists."
    )
    
    user_prompt = f"User Query: {query}\n\nResearch Findings:\n{findings}"
    
    try:
        validation = structured_llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        status = validation.status
    except Exception as e:
        # Graceful fallback if Groq API is down
        status = "sufficient"  # Default to sufficient to avoid infinite loops on failure

    return {
        "validation_result": status
    }
