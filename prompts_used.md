
---

**PROMPTS USED & DEVELOPMENT REASONING**
**Developer: Muhammad Ismail**

---

**PHASE 1 — SCAFFOLDING WITH AI ASSISTANCE**

I used Google Antigravity to generate the initial project boilerplate. The prompts I gave were intentionally high-level because I wanted the structure set up fast so I could focus my energy on the actual agent logic and graph routing, which is where the real complexity lives.

Scaffold prompt given to Google Antigravity:
"Create a Python project with this folder structure: src/agents/, src/graph/, src/tools/, app.py, main.py, requirements.txt, .env.example. Install langgraph, langchain, langchain-groq, tavily-python, streamlit, python-dotenv. Set up a config.py that loads API keys from environment variables."

State schema prompt:
"Define a LangGraph TypedDict state schema with these fields: messages using add_messages reducer, query, clarity_status, research_findings, confidence_score, attempts, validation_result, final_response. Add docstrings."

I reviewed both outputs before moving on and adjusted the state fields myself the original scaffold did not include attempts as an integer counter, which I added because I knew from reading the LangGraph docs that the validator loop needed a hard ceiling to prevent infinite retries.

---

**PHASE 2 — AGENT LOGIC (WHERE I TOOK OVER)**

This is where I stopped relying on AI generation and started thinking through the design myself.

The first real decision I made was how the Clarity Agent should determine ambiguity. My reasoning: a query is only "clear" if it contains a real company name AND a researchable intent. Vague queries like "tell me about that tech company" should always trigger the interrupt. I wrote the Clarity Agent system prompt myself with this logic and used claude to improve it, and iterated on the wording until the LLM was consistent.

The confidence scoring in the Research Agent was my design decision. I set the threshold at 6 out of 10 because I wanted a middle ground, too low and the validator becomes useless, too high and it retries on perfectly good results. The agent self-assesses based on how specific and concrete the Tavily results were.

The validator loop cap at 3 attempts was also my call. Without it, insufficient results on an obscure company would loop forever. After 3 attempts the system falls through to synthesis with whatever it has, and the synthesis agent is instructed to be transparent about data gaps.

---

**PHASE 3 — BUGS I FOUND AND FIXED MYSELF**

This is the part I am most proud of because these were not obvious bugs. They required actually understanding how LangGraph manages state across interrupt boundaries.

**Bug 1: Interrupt Resumption Loop**

After the Clarity Agent triggered a clarification interrupt and the user responded, the graph was resuming but immediately asking for clarification again. Infinite loop.

The root cause: when resuming the graph, I was updating state.query with the user's clarification text but not appending a new HumanMessage to state.messages. So the Clarity Agent would read the messages history, see the old ambiguous query as the last human message, and flag it as unclear again.

The fix I implemented: on resumption, call graph.update_state() with both the updated query AND the new HumanMessage appended to messages before calling graph.stream(None, config). The Clarity Agent now reads the last human message from messages rather than relying on the query field alone.

**Bug 2: Context Blindness on Follow-up Questions**

When a user asked "what about their competitors?" after researching OpenAI, the agents had no memory of the company. They were only reading the current query string, not the conversation history.

My fix: I updated the Clarity Agent prompt to explicitly scan the full messages history for any previously mentioned company name before deciding clarity status. If a company was established in a prior turn, follow-up questions inherit that context and are marked clear automatically. I also passed the last 4 messages as context into the Research Agent prompt so it knows what was already discussed.

**Bug 3: Streamlit Rerun Race Condition**

After the interrupt resumed, the UI was sometimes duplicating messages or showing stale state because Streamlit reruns the entire script on every interaction.

My fix: I added an awaiting_clarification boolean flag and a clarification_question string to st.session_state. When the interrupt fires, the flag is set. On the next user input, the app checks the flag first and routes to resume logic rather than a fresh graph invocation. The flag clears after resumption. This made the UI state deterministic.

---

**PHASE 5 — PRODUCTION CLEANUP PROMPT**

After the core logic was working and all three test scenarios passed, I gave Google Antigravity a final cleanup prompt to bring code quality to production standard:

"Do a full production-grade cleanup pass on the entire codebase. 
Add module-level and function-level docstrings to every file. 
Add type hints to every function signature. Replace all hardcoded strings with constants in config.py. 
Wrap all Tavily and Groq API calls in try/except with meaningful fallback behavior. 
Handle missing API keys in the UI with a clear Streamlit error. 
Ensure no business logic lives in app.py. 
Refactor any agent file over 80 lines so prompt construction is a separate helper function. Confirm requirements.txt reflects every import actually used."

I reviewed every change from this pass manually and reverted two refactors that broke the interrupt flow by moving state mutation logic into helper functions incorrectly.

---
