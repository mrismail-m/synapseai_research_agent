"""
State definition module for the LangGraph execution.

Defines the AgentState TypedDict which holds the conversation history,
intermediate research data, validation statuses, and final outputs.
"""

from typing import Annotated, TypedDict, Literal
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

class AgentState(TypedDict):
    """
    Represents the state of the research assistant graph.

    Attributes:
        messages (list[AnyMessage]): A list of messages in the conversation, managed by LangGraph's add_messages reducer.
        query (str): The initial user query or the current research goal.
        clarity_status (Literal["clear", "needs_clarification"]): Indicates if the user query is clear or needs further clarification.
        research_findings (str): Collected information and data from various research tools.
        confidence_score (int): An integer from 0 to 10 representing the assistant's confidence in the findings.
        validation_result (Literal["sufficient", "insufficient"]): Indicates whether the research findings are sufficient to answer the query.
        attempts (int): Number of research/validation loops performed to track progress and prevent infinite loops.
        final_response (str): The final comprehensive response to be delivered to the user.
    """
    messages: Annotated[list[AnyMessage], add_messages]
    query: str
    clarity_status: Literal["clear", "needs_clarification"]
    research_findings: str
    confidence_score: int
    validation_result: Literal["sufficient", "insufficient"]
    attempts: int
    final_response: str
