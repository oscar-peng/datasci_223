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

# Assignment 7: Clinical NLP with LLMs and Embeddings — Solution

## Setup

```python
%pip install -q -r requirements.txt

%reset -f
```

```python
import os
import json
import numpy as np
from dotenv import load_dotenv

load_dotenv()
print("Setup complete!")
```

---

## Part 1: Clinical Entity Extraction

```python
from extractor import get_client, build_prompt, call_llm, parse_json_response, validate_response, extract_entities
```

### `build_prompt`

```python
# Patch build_prompt into extractor module
import extractor

def build_prompt(note, few_shot=False):
    if few_shot:
        return f"""Extract structured medical data from clinical notes. Return JSON only.

Example:
Note: "65-year-old female with polyuria, polydipsia. Fasting glucose 285 mg/dL, HbA1c 9.2%.
Taking metformin 1000mg BID and lisinopril 10mg daily. Assessment: Poorly controlled type 2 diabetes mellitus."

Output:
{{
  "diagnosis": "Poorly controlled type 2 diabetes mellitus",
  "medications": ["metformin 1000mg BID", "lisinopril 10mg daily"],
  "lab_values": {{"fasting_glucose": "285 mg/dL", "HbA1c": "9.2%"}},
  "confidence": 0.95
}}

Now extract from this note:
Note: "{note}"

Output:"""

    return f"""Extract structured medical data from the following clinical note.
Return ONLY a JSON object with exactly these fields:

{{
  "diagnosis": "primary diagnosis as a string",
  "medications": ["list of medications with doses"],
  "lab_values": {{"test_name": "value with units"}},
  "confidence": 0.0
}}

Rules:
- confidence is a float from 0.0 to 1.0
- include ALL medications mentioned with doses if given
- include ALL lab values with units
- if a field has no data, use an empty list [] or empty dict {{}}

Clinical note:
{note}"""

extractor.build_prompt = build_prompt
```

### `parse_json_response`

```python
def parse_json_response(text):
    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences
    if "```" in text:
        start = text.find("```")
        end = text.rfind("```")
        if start != end:
            block = text[start:end]
            lines = block.split("\n")
            block = "\n".join(lines[1:])
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                pass

    # Find outermost braces
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    return None

extractor.parse_json_response = parse_json_response
```

### `validate_response`

```python
def validate_response(response):
    if not isinstance(response, dict):
        return False
    required = {"diagnosis", "medications", "lab_values", "confidence"}
    return required.issubset(response.keys())

extractor.validate_response = validate_response
```

### `extract_entities`

```python
def extract_entities(note, few_shot=False):
    client, provider = get_client()
    prompt = build_prompt(note, few_shot=few_shot)
    raw = call_llm(prompt, provider=provider, client=client)
    parsed = parse_json_response(raw)
    if parsed is None:
        return None
    if not validate_response(parsed):
        return None
    return parsed

extractor.extract_entities = extract_entities
```

### Test extraction on all 4 notes

```python
with open("clinical_notes.txt") as f:
    content = f.read()

sections = content.split("## Note")
notes = [s.split("\n", 1)[1].strip() for s in sections[1:] if s.strip()]
print(f"Loaded {len(notes)} notes\n")

for i, note in enumerate(notes, 1):
    result = extract_entities(note, few_shot=True)
    print(f"--- Note {i} ---")
    if result:
        print(json.dumps(result, indent=2))
    else:
        print("Extraction failed")
    print()
```

---

## Part 2: Semantic Search

```python
from search import load_notes, embed_notes, find_similar, save_results
import search
```

### `load_notes`

```python
def load_notes(filepath="clinical_notes.txt"):
    with open(filepath) as f:
        content = f.read()
    sections = content.split("## Note")
    notes = []
    for s in sections[1:]:
        text = s.split("\n", 1)
        if len(text) > 1:
            cleaned = text[1].strip()
            if cleaned:
                notes.append(cleaned)
    return notes

search.load_notes = load_notes
```

### `embed_notes`

```python
from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_notes(notes):
    return _model.encode(notes)

search.embed_notes = embed_notes
```

### `find_similar`

```python
from sklearn.metrics.pairwise import cosine_similarity

def find_similar(query, notes, embeddings, top_k=2):
    query_emb = _model.encode([query])
    scores = cosine_similarity(query_emb, embeddings)[0]
    ranked = sorted(
        zip(notes, scores),
        key=lambda x: x[1],
        reverse=True,
    )
    return [{"note": n, "score": float(s)} for n, s in ranked[:top_k]]

search.find_similar = find_similar
```

### `save_results`

```python
def save_results(results, filepath="search_results.json"):
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

search.save_results = save_results
```

### Run the full search pipeline

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

save_results(
    find_similar("heart attack symptoms", notes, embeddings, top_k=2),
    "search_results.json",
)
print("\nSaved search_results.json")
```

---

## Validation

```python
print("Run 'pytest .github/tests/ -v' in your terminal to check your work.")
```
