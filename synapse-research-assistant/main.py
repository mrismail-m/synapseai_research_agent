"""
Streamlit UI Module.

Handles the user interface, session state management, and real-time streaming
of LangGraph events for the Synapse Research Assistant. Contains zero business logic.
"""

import streamlit as st
import uuid
from typing import Optional
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command
from langgraph.pregel.types import StateSnapshot

# Page configuration MUST be the first Streamlit command
st.set_page_config(page_title="Synapse AI", page_icon="💡", layout="centered")

# Inject Custom Minimalist CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Minimalist background adapting to system theme */
.stApp {
    background-color: transparent;
}

/* Hide Streamlit components */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

/* Minimalist typography */
h1 {
    font-weight: 600 !important;
    font-size: 1.75rem !important;
    padding-bottom: 0.2rem;
    letter-spacing: -0.02em;
}

.subtitle {
    color: #737373;
    font-size: 0.95rem;
    font-weight: 400;
    margin-bottom: 2rem;
}

/* Clean chat bubble styling */
[data-testid="stChatMessage"] {
    background: transparent;
    border: none;
    border-radius: 0;
    padding: 1.5rem 0;
    margin-bottom: 0;
    border-bottom: 1px solid rgba(128, 128, 128, 0.1);
}

/* Remove default avatar styling if possible, make it blend */
.stChatMessageAvatar {
    opacity: 0.8;
}

/* Markdown text styling */
.stMarkdown p {
    line-height: 1.6;
    font-size: 0.95rem;
}

/* Input container - super clean */
.stChatInputContainer {
    background: transparent !important;
    border: 1px solid rgba(128, 128, 128, 0.2) !important;
    border-radius: 8px !important;
    padding: 0.25rem !important;
    box-shadow: none !important;
}

/* Status text */
.status-container {
    padding: 0.5rem 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

.status-text {
    color: #737373;
    font-weight: 400;
    font-size: 0.85rem;
    font-style: italic;
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

[data-testid="stChatMessage"] {
    animation: fadeIn 0.3s ease-out forwards;
}
</style>
""", unsafe_allow_html=True)

# Application Header
st.title("Synapse AI")
st.markdown("<div class='subtitle'>Minimalist corporate research analyst.</div>", unsafe_allow_html=True)

# Validation of environment variables
try:
    from src.config import config
    config.validate()
except Exception as e:
    # Safely handle missing API keys gracefully without traceback
    st.error(f"Configuration Error: {e}\nPlease check your .env file or environment variables.")
    st.stop()

# Import graph after config validation to ensure safe initialization
from src.graph.graph import app

# Session State for LangGraph to keep track of persistent conversation
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

graph_config = {"configurable": {"thread_id": st.session_state.thread_id}}

def get_current_state() -> Optional[StateSnapshot]:
    """
    Retrieves the current state of the LangGraph execution.
    
    Returns:
        StateSnapshot | None: The active graph state if present, else None.
    """
    try:
        return app.get_state(graph_config)
    except Exception:
        return None

state_snap = get_current_state()
messages = []
is_waiting = False

# Extract messages and routing status from persistent state
if state_snap:
    if "messages" in state_snap.values:
        messages = state_snap.values["messages"]
    # state.next contains pending nodes if the graph is paused (interrupted)
    if state_snap.next:
        is_waiting = True

# Display history
for msg in messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Input handling
if prompt := st.chat_input("Enter company or topic to research..."):
    # Display the user's message immediately for responsive UI
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Placeholder for status indicator
    status_placeholder = st.empty()
    
    try:
        if is_waiting:
            # INTERRUPT RESUMPTION FLOW:
            # 1. Manually mutate state to inject the user's new message
            app.update_state(graph_config, {"messages": [HumanMessage(content=prompt)], "query": prompt})
            # 2. Issue a Command to resume graph execution from the suspended state
            stream = app.stream(Command(resume=True), graph_config, stream_mode="updates")
        else:
            # Initial invocation of the graph
            inputs = {
                "messages": [HumanMessage(content=prompt)],
                "query": prompt
            }
            stream = app.stream(inputs, graph_config, stream_mode="updates")
        
        # Listen to the graph stream for status updates (no business logic, just UI updates)
        for event in stream:
            for node_name, node_state in event.items():
                status_msg = ""
                if node_name == "clarity_agent":
                    status_msg = "Evaluating request..."
                elif node_name == "research_agent":
                    status_msg = "Gathering intelligence..."
                elif node_name == "validator_agent":
                    status_msg = "Validating findings..."
                elif node_name == "synthesis_agent":
                    status_msg = "Synthesizing briefing..."
                elif node_name == "clarification_point":
                    status_msg = "Awaiting clarification..."
                
                if status_msg:
                    status_placeholder.markdown(f"""
                        <div class='status-container'>
                            <div class='status-text'>○ {status_msg}</div>
                        </div>
                    """, unsafe_allow_html=True)

        # Clear status indicator once finished
        status_placeholder.empty()
        
        # Trigger full UI refresh to cleanly re-render the fully-populated state history
        st.rerun()

    except Exception as e:
        status_placeholder.empty()
        st.error(f"Agent Execution Error: {str(e)}")
