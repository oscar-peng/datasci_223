# Assignment 7: Clinical NLP with LLMs and Embeddings

Extract structured data from clinical notes using LLM prompt engineering, then build a semantic search system using sentence embeddings.

**Dataset**: 75 synthetic discharge summaries from [Asclepius-Synthetic-Clinical-Notes](https://huggingface.co/datasets/aisc-team-a1/Asclepius-Synthetic-Clinical-Notes) (Kweon et al., 2023) in `asclepius_notes.json`. Part 2 also uses 4 curated notes in `clinical_notes.txt`.

## Getting Started

```bash
pip install -r requirements.txt
```

### API Key (Part 1 only)

Part 1 requires an LLM API key. We use [OpenRouter](https://openrouter.ai) (OpenAI-compatible):

1. Sign up at [openrouter.ai](https://openrouter.ai)
2. Create an API key under "Keys"
3. Copy the example env file and fill in your key:
   ```bash
   cp example.env .env
   ```

OpenAI keys also work — set `OPENAI_API_KEY` in `.env` instead.

Part 2 (embeddings) runs locally and does not need an API key.

## Workflow

Open `assignment.md` as a Jupyter notebook (convert with `jupytext --to notebook assignment.md` if needed) and work through both parts:

1. **Part 1: Clinical Entity Extraction** — Implement `build_prompt`, `parse_json_response`, `validate_response`, and `extract_entities`
2. **Part 2: Semantic Search** — Implement `load_notes`, `embed_notes`, `find_similar`, and `save_results`

The notebook includes "do not modify" save cells that write your implementations to `extractor.py` and `search.py` for autograding.

## Output Files

| File | Part | Description |
|:---|:---|:---|
| `extractor.py` | 1 | Entity extraction functions (saved by notebook) |
| `search.py` | 2 | Semantic search functions (saved by notebook) |
| `search_results.json` | 2 | JSON output from semantic search |

## Checking Your Work

```bash
# Run all tests
python -m pytest .github/tests/ -v

# Run tests for a specific part
python -m pytest .github/tests/test_extractor.py -v
python -m pytest .github/tests/test_search.py -v
```

Note: Extractor tests do **not** make live API calls — they test prompt building, JSON parsing, and validation only. Search tests run the embedding model locally.

## Hints

See `hints.md` for troubleshooting tips.
