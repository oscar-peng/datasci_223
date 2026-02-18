---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Assignment 7: Clinical NLP with LLMs and Embeddings

Extract structured data from clinical notes using LLM prompt engineering, then build a semantic search system using sentence embeddings.

**Dataset:** 75 synthetic discharge summaries from [Asclepius-Synthetic-Clinical-Notes](https://huggingface.co/datasets/aisc-team-a1/Asclepius-Synthetic-Clinical-Notes) (Kweon et al., 2023) in `asclepius_notes.json`. Part 2 also uses 4 curated notes in `clinical_notes.txt`.

## Setup

```python
%pip install -q -r requirements.txt

%reset -f
```

```python
import os
import json
import random
import numpy as np
from dotenv import load_dotenv

load_dotenv()
print("Setup complete!")
```

### API Key

Part 1 requires an [OpenRouter](https://openrouter.ai) API key (OpenAI keys also work). Copy the example and fill in your key:

```bash
cp example.env .env
# Then edit .env with your actual key
```

Part 2 runs locally and does not need an API key.

### Load Asclepius Notes

```python
with open("asclepius_notes.json") as f:
    asclepius = json.load(f)

print(f"Loaded {len(asclepius)} synthetic clinical notes from Asclepius")
print(f"Keys: {list(asclepius[0].keys())}")
```

```python
# Preview a sample note
print(asclepius[0]["note"][:500] + "...")
```

```python
# Select 4 notes for entity extraction (Part 1)
random.seed(2026)
sample = random.sample(asclepius, 4)
notes_p1 = [s["note"] for s in sample]

print(f"Selected {len(notes_p1)} notes for extraction")
for i, n in enumerate(notes_p1, 1):
    print(f"\n--- Note {i} ({len(n)} chars) ---")
    print(n[:150] + "...")
```

---

## Part 1: Clinical Entity Extraction

Implement functions to extract structured medical data from clinical notes using LLM prompt engineering.

`extractor.py` provides two functions already:
- `get_client()` — initializes the OpenRouter/OpenAI client
- `call_llm(prompt, provider, client)` — sends a prompt and returns the response

You'll implement the remaining four functions below.

```python
from extractor import get_client, call_llm
import extractor
```

### `build_prompt`

Build a prompt that instructs the LLM to extract structured data from a clinical note.

```python
# TODO: Implement build_prompt
# Requirements:
#   - Describe the extraction task clearly
#   - Specify the JSON output schema with these fields:
#     {"diagnosis": str, "medications": list, "lab_values": dict, "confidence": float}
#   - When few_shot=True, include 1-2 example input/output pairs
#   - Include the clinical note text
def build_prompt(note, few_shot=False):
    pass  # replace with your implementation

extractor.build_prompt = build_prompt
```

### `parse_json_response`

Extract a JSON object from LLM response text, which may contain markdown code fences or other wrapping.

```python
# TODO: Implement parse_json_response
# Requirements:
#   - Handle clean JSON strings (direct json.loads)
#   - Handle JSON wrapped in ```json ... ``` markdown blocks
#   - Find JSON within surrounding text (look for outermost { and })
#   - Return None if parsing fails
def parse_json_response(text):
    pass  # replace with your implementation

extractor.parse_json_response = parse_json_response
```

### `validate_response`

Check that a parsed response dict contains all required keys.

```python
# TODO: Implement validate_response
# Required fields: diagnosis, medications, lab_values, confidence
# Return True if all present, False otherwise
def validate_response(response):
    pass  # replace with your implementation

extractor.validate_response = validate_response
```

### `extract_entities`

Orchestrate the full extraction pipeline: get client, build prompt, call LLM, parse, validate, return.

```python
# TODO: Implement extract_entities
# Steps:
#   1. client, provider = get_client()
#   2. prompt = build_prompt(note, few_shot=few_shot)
#   3. raw = call_llm(prompt, provider=provider, client=client)
#   4. parsed = parse_json_response(raw)
#   5. Validate and return (return None if parsing or validation fails)
def extract_entities(note, few_shot=False):
    pass  # replace with your implementation

extractor.extract_entities = extract_entities
```

### Test extraction

```python
for i, note in enumerate(notes_p1, 1):
    result = extract_entities(note, few_shot=True)
    print(f"--- Note {i} ---")
    if result:
        print(json.dumps(result, indent=2))
    else:
        print("Extraction failed")
    print()
```

```python
# Save implementations to extractor.py for autograding (do not modify this cell)
import inspect as _insp

_parts = [
    '"""\nLLM Prompt Engineering Assignment: Clinical Entity Extraction\n\n'
    "Complete the functions below to extract structured data from clinical notes\n"
    'using LLM APIs.\n"""\n\n'
    "import json\nimport os\nfrom typing import Optional\n",
]

for _fn in [get_client, build_prompt, call_llm, extract_entities, validate_response, parse_json_response]:
    _parts.append("\n\n" + _insp.getsource(_fn))

with open("extractor.py", "w") as _f:
    _f.write("".join(_parts) + "\n")

print("Saved extractor.py")
```

---

## Part 2: Semantic Search

Build a semantic search system that finds clinical notes by meaning rather than keywords, using sentence embeddings and cosine similarity.

This part runs locally — no API key needed.

```python
from search import get_device
import search

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

_model = SentenceTransformer("all-MiniLM-L6-v2", device=get_device())
print(f"Model loaded on {get_device()}")
```

### `load_notes`

Parse `clinical_notes.txt` into a list of note strings.

```python
# TODO: Implement load_notes
# Requirements:
#   - Read the file contents
#   - Split on "## Note" headers
#   - Strip whitespace, skip empty strings
#   - The first split element is the file header — skip it
#   - Return a list of note text strings
def load_notes(filepath="clinical_notes.txt"):
    pass  # replace with your implementation

search.load_notes = load_notes
```

### `embed_notes`

Generate embeddings for a list of notes using the sentence transformer model.

```python
# TODO: Implement embed_notes
# Use _model.encode(notes) — returns a numpy array of shape (n_notes, embedding_dim)
def embed_notes(notes):
    pass  # replace with your implementation

search.embed_notes = embed_notes
```

### `find_similar`

Search notes by meaning using cosine similarity.

```python
# TODO: Implement find_similar
# Steps:
#   1. Embed the query with _model.encode([query])
#   2. Compute cosine_similarity(query_embedding, embeddings)
#   3. Sort by score descending
#   4. Return top_k results as [{"note": str, "score": float}, ...]
def find_similar(query, notes, embeddings, top_k=2):
    pass  # replace with your implementation

search.find_similar = find_similar
```

### `save_results`

Write search results to a JSON file.

```python
# TODO: Implement save_results
def save_results(results, filepath="search_results.json"):
    pass  # replace with your implementation

search.save_results = save_results
```

### Run the search pipeline

```python
notes = load_notes("clinical_notes.txt")
print(f"Loaded {len(notes)} notes")

embeddings = embed_notes(notes)
print(f"Embeddings: {embeddings.shape}")

queries = [
    "heart attack symptoms",
    "infectious disease with fever",
    "respiratory illness",
]

for q in queries:
    print(f"\nQuery: '{q}'")
    results = find_similar(q, notes, embeddings, top_k=2)
    for i, r in enumerate(results, 1):
        print(f"  {i}. (score: {r['score']:.3f}) {r['note'][:80]}...")
```

```python
# Save results (do not modify this cell)
save_results(
    find_similar("heart attack symptoms", notes, embeddings, top_k=2),
    "search_results.json",
)
print("Saved search_results.json")
```

```python
# Save implementations to search.py for autograding (do not modify this cell)
import inspect as _insp

_parts = [
    '"""\nSemantic Search Assignment: Clinical Note Search with Embeddings\n\n'
    "Use sentence embeddings to search clinical notes by meaning rather than keywords.\n"
    '"""\n\n'
    "import json\nimport numpy as np\nfrom typing import List, Dict\n",
]

_parts.append("\n\n" + _insp.getsource(get_device))
_parts.append('\n\nfrom sentence_transformers import SentenceTransformer\n')
_parts.append('from sklearn.metrics.pairwise import cosine_similarity\n\n')
_parts.append('_model = SentenceTransformer("all-MiniLM-L6-v2", device=get_device())\n')

for _fn in [load_notes, embed_notes, find_similar, save_results]:
    _parts.append("\n\n" + _insp.getsource(_fn))

with open("search.py", "w") as _f:
    _f.write("".join(_parts) + "\n")

print("Saved search.py")
```

---

## Validation

```python
print("Run 'pytest .github/tests/ -v' in your terminal to check your work.")
```
