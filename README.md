# AI Data Modeller

A Streamlit-based agentic AI application that analyzes uploaded CSV data to infer schema, detect relationships, generate SQL DDL, and produce ER diagram specifications using a ReAct agent with LangChain.

## Features

- Upload single or multiple CSV files
- Multi-provider LLM support: GROQ, OpenAI, Ollama
- ReAct agent orchestrates tool usage for analysis
- Primary key detection, composite key search, relationship discovery
- Generates SQL DDL statements
- Produces ER diagram specification (JSON) and renders a Graphviz ERD
- Clean Streamlit UI with tabs: Summary, SQL DDL, ER Diagram, Agent Process
- Download buttons for SQL DDL, ER JSON, and DOT

## What’s New

- The app now prefers the agent’s Final Answer for SQL DDL and ER JSON.
  - SQL is taken from a fenced ```sql block (or from the first CREATE TABLE up to the next section if no fence).
  - ER spec is taken from a fenced ```json block (with a safe fallback parser).
- ER data is normalized for the UI (so entities can be a dict or a list from the agent output).
- Guardrails added to avoid PromptTemplate brace errors and early agent stopping.

## Project Structure

- `streamlit_app.py` — Streamlit frontend
- `intelligent_schema_analyzer.py` — ReAct agent and analysis logic (Final Answer parsing, normalization, fallbacks)
- `tool.py` — Data tools: PK/composite key detection, relationship finder, ERD DOT generator
- `requirements.in` — Source list of dependencies
- `requirements.txt` — Pinned lock file (compiled from `requirements.in`)

## Requirements

- Python 3.10+
- Conda environment (recommended)
- Optional: Ollama running locally for local models

## Setup

```bash
# Create and activate environment (example with conda)
conda create -n AiDataModeller python=3.10 -y
conda activate AiDataModeller

# Install dependencies
pip install uv
uv pip install -r requirements.txt
```

If you only have `requirements.in`, compile it first:

```bash
uv pip compile requirements.in -o requirements.txt
uv pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root:

```
GROQ_API_KEY="<your-groq-key>"
OPENAI_API_KEY="<your-openai-key>"
HF_TOKEN="<optional-hf-token>"
```

Security: Do not commit real API keys. Add `.env` to `.gitignore` and rotate any exposed keys.

## Run the App

```bash
conda activate AiDataModeller
streamlit run streamlit_app.py
```

Open http://localhost:8501 and:
- Upload CSVs or click "Use Sample Data"
- Choose provider (GROQ/OpenAI/Ollama) and model
- Initialize Analyzer
- Run Analysis

## How It Works

- The ReAct agent (LangChain) uses tools from `tool.py` to:
  - Analyze primary keys (`analyze_primary_key_candidates`)
  - Find composite keys (`find_composite_keys`)
  - Detect relationships (`find_dataframe_relations`)
- `intelligent_schema_analyzer.py`:
  - Prompts the agent to return a Final Answer with fenced blocks.
  - Parses SQL (```sql … ```) and ER JSON (```json … ```), with tool-based fallbacks.
  - Normalizes ER structures for UI consumption.

## Providers & Models

- GROQ: `llama3-8b-8192`, `llama3-70b-8192`, `mixtral-8x7b-32768`
- OpenAI: `gpt-3.5-turbo`, `gpt-4`, `gpt-4-turbo`, `gpt-5-preview`
- Ollama: `llama3`, `codellama`, `phi3`, `mistral` (requires Ollama installed and the model pulled)

## Troubleshooting

- PromptTemplate error mentioning `{ "entities" }`:
  - Cause: braces in prompt treated as variables. Fixed by removing inline JSON examples and using guidance text.
- Agent stopped due to iteration/time limit:
  - The agent has increased limits and a prompt nudge. If it persists, try a faster/more capable model or fewer/lighter CSVs.
- SQL DDL tab shows extra ER text:
  - Extraction refined: only fenced ```sql or the block from first CREATE TABLE up to the next section is shown.
- ER Diagram tab errors (entities as list):
  - ER spec is normalized (both backend and UI). If custom outputs are plugged in, ensure top-level `entities` is a dict mapping entity name to details.
- JSON download serialization errors:
  - A custom serializer handles NumPy/Pandas types in the UI.

## Development Notes

- Follow PEP8 and type hints.
- Keep tools in `tool.py`, orchestrate in `intelligent_schema_analyzer.py`.
- Prefer simple, modular implementations.
- Tests: add pytest unit tests for tool functions.

## License

MIT
