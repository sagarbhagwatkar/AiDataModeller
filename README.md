# AI Data Modeller

A Streamlit-based agentic AI application that analyzes uploaded CSV data to infer schema, detect relationships, generate SQL DDL, and produce ER diagram specifications using a ReAct agent with LangChain.

## Features

- Upload single or multiple CSV files
- Multi-provider LLM support: GROQ, OpenAI, Ollama
- ReAct agent orchestrates tool usage for analysis
- Primary key detection, composite key search, relationship discovery
- Generates SQL DDL statements
- Produces ER diagram specification (JSON) and Graphviz DOT (via tool)
- Clean Streamlit UI with tabs: Summary, SQL DDL, ER Diagram, Agent Process

## Project Structure

- `streamlit_app.py` — Streamlit frontend
- `intelligent_schema_analyzer.py` — ReAct agent and analysis logic
- `tool.py` — Data tools: PK/composite key detection, relationship finder, ERD DOT generator
- `requirements.in` — Source list of dependencies
- `requirements.txt` — Pinned lock file (compiled from `requirements.in`)

## Requirements

- Python 3.10+
- Conda environment (recommended)
- Optional: Running Ollama for local models

## Setup

```bash
# Create and activate environment (example with conda)
conda create -n AiDataModeller python=3.10 -y
conda activate AiDataModeller

# Install dependencies (uv recommended for speed)
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
GROQ_API_KEY="..."
OPENAI_API_KEY="..."
HF_TOKEN="..."  # optional
```

Note: Keep keys private. Do not commit `.env` to version control.

## Run the App

```bash
conda activate AiDataModeller
streamlit run streamlit_app.py
```

Open http://localhost:8502 and:
- Upload CSVs or click "Use Sample Data"
- Choose provider (GROQ/OpenAI/Ollama) and model
- Initialize Analyzer
- Run Analysis

## How It Works

- The ReAct agent (LangChain) uses tools from `tool.py` to:
  - Analyze primary keys (`analyze_primary_key_candidates`)
  - Find composite keys (`find_composite_keys`)
  - Detect relationships (`find_dataframe_relations`)
- `intelligent_schema_analyzer.py` aggregates results, generates SQL DDL and ER specs, and exposes them to the UI.

## Testing Tools Quickly

```bash
python - <<'PY'
import pandas as pd
from tool import analyze_primary_key_candidates, find_composite_keys, find_dataframe_relations

frames = {
  'customers': pd.DataFrame({'customer_id':[1,2,3], 'name':['a','b','c']}),
  'orders': pd.DataFrame({'order_id':[10,11,12], 'customer_id':[1,1,2]})
}
print('PK:', analyze_primary_key_candidates(frames))
print('Composite:', find_composite_keys(frames))
print('Rels:', find_dataframe_relations(frames))
PY
```

## Notes & Tips

- GROQ is fast; OpenAI can be more capable; Ollama is local/private.
- For large CSVs, consider sampling before upload.
- If you see JSON serialization errors in downloads, the UI uses a custom serializer to handle NumPy/Pandas types.

## License

MIT
