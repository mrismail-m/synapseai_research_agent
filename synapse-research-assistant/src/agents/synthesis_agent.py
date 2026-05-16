"""
Synthesis Agent Module.

Takes the raw research findings and synthesizes them into a final, polished 
report tailored for a C-suite executive audience.
"""

from typing import Dict, Any, List
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, AnyMessage
from src.config import config
from src.graph.state import AgentState

def _build_synthesis_prompt(messages: List[AnyMessage], query: str, findings: str) -> tuple[str, str]:
    """Helper to build the system and user prompts for the synthesis agent."""
    system_prompt = (
        "You are a senior research analyst writing for C-suite executives. "
        "Produce a structured briefing based strictly on the research findings provided.\n\n"
        "RULES:\n"
        "1. Answer only what was asked. If the question is about competitors, write only about competitors. "
        "Do not add leadership, financials, or history unless explicitly requested.\n"
        "2. Use '### Header' (Markdown H3) to separate sections. Do not use bold (**) for headers.\n"
        "3. No emojis. No filler phrases like 'it is worth noting' or 'in conclusion'. No marketing language.\n"
        "4. If a data point is missing, omit it. Never write 'data unavailable' or use placeholders.\n"
        "5. Ground every claim in the research findings. Do not infer or hallucinate beyond what was retrieved.\n"
        "6. If the user asked a follow-up question, answer it in the context of the prior conversation. "
        "Do not re-summarize the company from scratch.\n\n"
        "FORMAT:\n"
        "Open with one sentence stating what this briefing covers. "
        "Then use sections with '### Header'. "
        "Close with a 2-sentence takeaway if the findings support a clear conclusion."
    )
    
    # We pass history to ensure context awareness in the final writing
    history_context = "\n".join([f"{m.type}: {m.content}" for m in messages[-5:]])
    
    user_prompt = (
        f"Conversation History:\n{history_context}\n\n"
        f"Original User Request: {query}\n\n"
        f"Raw Research Findings:\n{findings}\n\n"
        "Please synthesize this into a direct, high-level report."
    )
    return system_prompt, user_prompt

def synthesis_agent(state: AgentState) -> Dict[str, Any]:
    """
    Synthesizes research findings into a professional, direct report.
    Forbids emojis and focuses on straightforward information.
    
    Args:
        state (AgentState): The current graph state.
        
    Returns:
        dict: State updates containing the final response and the message to append.
    """
    llm = ChatGroq(
        model=config.LLM_MODEL_NAME,
        api_key=config.GROQ_API_KEY,
        temperature=config.LLM_TEMPERATURE_CREATIVE
    )
    
    findings = state.get("research_findings", "")
    query = state.get("query", "")
    messages = state.get("messages", [])
    
    system_prompt, user_prompt = _build_synthesis_prompt(messages, query, findings)
    
    try:
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
    except Exception as e:
        # Graceful fallback if Groq API is unreachable
        fallback_text = f"Synthesis failed due to an API error: {str(e)}. Raw findings:\n\n{findings}"
        response = AIMessage(content=fallback_text)
    
    return {
        "final_response": response.content,
        "messages": [response] # This adds the AIMessage to history
    }
