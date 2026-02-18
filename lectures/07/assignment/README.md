# Assignment 7: Clinical NLP with LLMs and Embeddings

Extract structured data from clinical notes using LLM prompt engineering, then build a semantic search system using sentence embeddings.

**Dataset**: 75 synthetic discharge summaries from [Asclepius-Synthetic-Clinical-Notes](https://huggingface.co/datasets/aisc-team-a1/Asclepius-Synthetic-Clinical-Notes) (Kweon et al., 2023) in `asclepius_notes.json`.

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
2. **Part 2: Semantic Search** — Implement `embed_notes` and `find_similar`

The notebook saves results to `output/` for autograding.

## Output Files

| File | Part | Description |
|:---|:---|:---|
| `output/extraction_results.json` | 1 | Structured entities extracted from clinical notes |
| `output/search_results.json` | 2 | Semantic search results with similarity scores |

## Checking Your Work

```bash
python -m pytest .github/tests/ -v
```

Note: Tests check output artifacts only — run the notebook first, then run tests.

## Hints

See `hints.md` for troubleshooting tips.
