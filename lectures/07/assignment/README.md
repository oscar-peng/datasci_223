# Assignment 7: Clinical NLP with LLMs and Embeddings

Extract structured data from clinical notes using LLM prompt engineering, then build a semantic search system using sentence embeddings.

**Dataset**: 4 clinical notes in `clinical_notes.txt` covering STEMI, diabetes, pneumonia, and bacterial meningitis.

## Learning Objectives

- Build effective prompts for structured data extraction (zero-shot and few-shot)
- Parse and validate JSON responses from LLM APIs
- Generate sentence embeddings from clinical text
- Implement cosine-similarity-based semantic search

## Setup

```bash
pip install -r requirements.txt
```

### API Key (Part 1 only)

Part 1 requires an LLM API key. We use [OpenRouter](https://openrouter.ai) (OpenAI-compatible):

1. Sign up at [openrouter.ai](https://openrouter.ai)
2. Create an API key under "Keys"
3. Set it as an environment variable:
   ```bash
   export OPENROUTER_API_KEY=your_key_here
   ```
   Or create a `.env` file: `OPENROUTER_API_KEY=your_key_here`

OpenAI keys also work — set `OPENAI_API_KEY` instead.

Part 2 (embeddings) runs locally and does not need an API key.

## Your Tasks

### Part 1: Clinical Entity Extraction (`extractor.py`)

Complete the TODO functions in `extractor.py` to extract structured medical data from clinical notes.

**What to implement:**

1. **`build_prompt(note, few_shot=False)`** — Build a prompt that:
    - Describes the extraction task clearly
    - Specifies the expected JSON output schema (diagnosis, medications, lab_values, confidence)
    - When `few_shot=True`, includes 1-2 example input/output pairs before the target note
    - Includes the clinical note text

2. **`parse_json_response(response_text)`** — Extract JSON from LLM response text:
    - Handle clean JSON strings
    - Handle JSON wrapped in `` ```json ... ``` `` markdown blocks
    - Return `None` if parsing fails

3. **`validate_response(response)`** — Check that a parsed response dict contains all required keys:
    - `diagnosis` (str)
    - `medications` (list)
    - `lab_values` (dict)
    - `confidence` (float, 0-1)
    - Return `True` if valid, `False` otherwise

4. **`extract_entities(note, few_shot=False)`** — Orchestrate the full pipeline:
    - Get the LLM client
    - Build the prompt
    - Call the LLM
    - Parse the JSON response
    - Validate and return the result

**Test your work:**
```bash
python extractor.py
```

### Part 2: Semantic Search (`search.py`)

Create `search.py` to embed clinical notes and search them by meaning.

**What to implement:**

1. **`load_notes(filepath)`** — Parse `clinical_notes.txt` into a list of note strings
    - Split on `## Note` headers
    - Strip whitespace, skip empty strings
    - Return a list of note text strings

2. **`embed_notes(notes)`** — Generate embeddings for a list of notes
    - Use `SentenceTransformer("all-MiniLM-L6-v2")`
    - Return a numpy array of shape `(n_notes, embedding_dim)`

3. **`find_similar(query, notes, embeddings, top_k=2)`** — Semantic search
    - Embed the query using the same model
    - Compute cosine similarity against all note embeddings
    - Return a list of dicts: `[{"note": str, "score": float}, ...]` sorted by score descending
    - Limit to `top_k` results

4. **`save_results(results, filepath="search_results.json")`** — Write results to JSON file

**Test your work:**
```bash
python search.py
```

Running `search.py` should:
- Load the 4 clinical notes
- Embed them
- Run a sample query (e.g., "heart attack symptoms")
- Print results and save to `search_results.json`

## Output Files

| File | Part | Description |
|:---|:---|:---|
| `extractor.py` | 1 | Completed entity extraction functions |
| `search.py` | 2 | Semantic search implementation |
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
