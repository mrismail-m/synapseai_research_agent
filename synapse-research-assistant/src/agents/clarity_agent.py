"""
Clarity Agent Module.

Evaluates user queries to determine if they are specific enough to proceed
with research, or if they require clarification from the user.
"""

from typing import Literal, Dict, Any
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from src.config import config
from src.graph.state import AgentState

class ClarityEvaluation(BaseModel):
    """Evaluation of the user query for research readiness."""
    status: Literal["clear", "needs_clarification"] = Field(
        description="Whether the query is specific enough to research a company."
    )
    clarification_message: str = Field(
        description="Message to the user if more information is needed. Empty if status is 'clear'."
    )

def clarity_agent(state: AgentState) -> Dict[str, Any]:
    """
    Evaluates the user's query for clarity and company name presence.
    Strictly follows conversation context to avoid unnecessary interrupts.
    
    Args:
        state (AgentState): The current state of the graph.
        
    Returns:
        dict: Updates to apply to the state, including the clarity status,
            the potentially modified query, and any clarification messages.
    """
    llm = ChatGroq(
        model=config.LLM_MODEL_NAME,
        api_key=config.GROQ_API_KEY,
        temperature=config.LLM_TEMPERATURE_STRICT
    )
    
    structured_llm = llm.with_structured_output(ClarityEvaluation)
    
    system_prompt = (
        "You are a Gatekeeper for a research graph. Your ONLY job is to decide if we have a subject "
        "to research. \n\n"
        "STRICT PASS/FAIL RULES:\n"
        "1. SUBJECT IN CONTEXT: If the history already mentions a company or product (like OpenAI, Claude, "
        "Anthropic, etc.), ANY follow-up question is 'clear'. DO NOT ask for more detail.\n"
        "2. NO ADVISING: Do not ask the user to specify 'which' competitors or 'what kind' of financials. "
        "If they ask for 'competitors', that is a clear intent. Mark it 'clear'.\n"
        "3. PRONOUNS: 'their', 'its', 'they' always refer to the subject already discussed.\n"
        "4. PASS UNLESS BLANK: If there is a subject in context, you must return status='clear'.\n\n"
        "If you see 'competitors' and the history mentions 'Claude', the subject is 'Claude's competitors'. "
        "This is CLEAR. Mark it 'clear'."
    )
    
    messages = state.get("messages", [])
    
    # Try analyzing the history with the LLM, use fallback if Groq is unreachable
    try:
        evaluation = structured_llm.invoke([
            {"role": "system", "content": system_prompt},
        ] + messages)
    except Exception as e:
        # Graceful fallback if the API is unreachable
        return {
            "clarity_status": "needs_clarification",
            "messages": [AIMessage(content=f"I'm having trouble connecting to my analysis engine. Could you clarify your request? (Error: {str(e)})")]
        }
    
    updates: Dict[str, Any] = {
        "clarity_status": evaluation.status
    }
    
    # Extract the last human message to serve as the active query
    if evaluation.status == "clear":
        last_human = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
        updates["query"] = last_human
        
    if evaluation.status == "needs_clarification":
        # Mutate state by adding the clarification request from the agent
        updates["messages"] = [AIMessage(content=evaluation.clarification_message)]
        
    return updates
