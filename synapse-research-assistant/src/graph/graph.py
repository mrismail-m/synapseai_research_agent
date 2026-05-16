"""
Graph Module.

Defines the LangGraph workflow, including nodes, conditional edges, 
and the compilation of the application with persistent memory.
"""

from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from src.config import config
from src.graph.state import AgentState
from src.agents.clarity_agent import clarity_agent
from src.agents.research_agent import research_agent
from src.agents.validator_agent import validator_agent
from src.agents.synthesis_agent import synthesis_agent

# --- Routing Logic ---

def route_after_clarity(state: AgentState) -> Literal["research_agent", "clarification_point"]:
    """
    Routes based on the clarity of the user query.
    
    If the clarity_agent determined the query needs clarification, routes to the
    interrupt node. Otherwise, proceeds to research.
    """
    if state.get("clarity_status") == "needs_clarification":
        return "clarification_point"
    return "research_agent"

def route_after_research(state: AgentState) -> Literal["synthesis_agent", "validator_agent"]:
    """
    Routes based on research confidence.
    
    If confidence meets the threshold defined in config, skips validation
    and goes straight to synthesis. Otherwise, triggers validation.
    """
    if state.get("confidence_score", 0) >= config.CONFIDENCE_THRESHOLD:
        return "synthesis_agent"
    return "validator_agent"

def route_after_validator(state: AgentState) -> Literal["research_agent", "synthesis_agent"]:
    """
    Routes based on validation result and attempt count.
    
    Loops back to research_agent if findings are insufficient and max attempts
    have not been reached. Otherwise, moves to synthesis.
    """
    if state.get("validation_result") == "insufficient" and state.get("attempts", 0) < config.MAX_RESEARCH_ATTEMPTS:
        return "research_agent"
    return "synthesis_agent"

# --- Graph Assembly ---

def wait_for_user_clarification(state: AgentState) -> AgentState:
    """
    Dummy node to handle the Human-in-the-Loop (HITL) interrupt.
    
    When the graph reaches here, it triggers an interrupt, suspending execution.
    Upon resumption by the client (via Command(resume=True)), it returns the state 
    and transitions back to the clarity_agent.
    """
    interrupt("Please provide more information about the company you'd like to research.")
    return state

# Initialize the state graph
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("clarity_agent", clarity_agent)
workflow.add_node("clarification_point", wait_for_user_clarification)
workflow.add_node("research_agent", research_agent)
workflow.add_node("validator_agent", validator_agent)
workflow.add_node("synthesis_agent", synthesis_agent)

# Set Entry Point
workflow.set_entry_point("clarity_agent")

# Add Conditional Edges
workflow.add_conditional_edges(
    "clarity_agent",
    route_after_clarity,
    {
        "clarification_point": "clarification_point",
        "research_agent": "research_agent"
    }
)

# After resumption from clarification_point, go back to clarity_agent
workflow.add_edge("clarification_point", "clarity_agent")

workflow.add_conditional_edges(
    "research_agent",
    route_after_research,
    {
        "synthesis_agent": "synthesis_agent",
        "validator_agent": "validator_agent"
    }
)

workflow.add_conditional_edges(
    "validator_agent",
    route_after_validator,
    {
        "research_agent": "research_agent",
        "synthesis_agent": "synthesis_agent"
    }
)

workflow.add_edge("synthesis_agent", END)

# Compile with MemorySaver for persistence across interrupts
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
