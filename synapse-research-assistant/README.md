# Synapse Research Assistant

An AI-powered research assistant built with LangGraph, LangChain, and Groq.

## Project Structure

- `src/agents/`: Agent logic and definitions.
- `src/graph/`: LangGraph workflow definitions.
- `src/tools/`: Custom tools for the assistant.
- `main.py`: CLI entry point.
- `app.py`: Streamlit web interface.

## Setup

1. Clone the repository.
2. Create a `.env` file from `.env.example` and add your API keys.
3. Set up a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the App

```bash
streamlit run main.py
```
