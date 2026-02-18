# Lecture 08 Demos: LLM Applications & Workflows

## Setup

```bash
pip install -r requirements.txt
```

### API Key

Demos 1-3 require an LLM API key. Copy the example and add your key:

```bash
cp example.env .env
# Edit .env with your OPENROUTER_API_KEY or OPENAI_API_KEY
```

### Convert to Notebooks

```bash
jupytext --to notebook *.md
```

## Demos

| Demo | File | Topics |
|:---|:---|:---|
| 1 | `01_rag_mcp.md` | RAG pipeline, ChromaDB, embeddings, MCP tool discovery |
| 2 | `02_agentic_workflow.md` | Prompt chaining, PHI guardrails, tool calling, Agents SDK |
| 3 | `03_failure_modes.md` | Hallucination, prompt injection, inconsistency, math errors, context overflow |

## Requirements

- Python 3.11+
- API key (OpenRouter or OpenAI)
- Node.js/npx (optional, for MCP server demo in Demo 1)
