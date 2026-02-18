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

# Assignment 8: LLM Applications — RAG & Guardrails

Build PHI guardrails and a RAG pipeline for clinical guideline Q&A.

**Guidelines data:** `sample_documents/guidelines.txt` — synthetic clinical guidelines covering hypertension, diabetes, and chest pain evaluation.

## Setup

```python
%pip install -q -r requirements.txt

%reset -f
```

```python
import os
import re
import json
import numpy as np
from dotenv import load_dotenv

os.makedirs("output", exist_ok=True)
load_dotenv()
print("Setup complete!")
```

### API Key

Part 2 requires an [OpenRouter](https://openrouter.ai) API key (OpenAI keys also work). Copy the example and fill in your key:

```bash
cp example.env .env
# Then edit .env with your actual key
```

Part 1 runs locally and does not need an API key.

### Helper Functions (do not modify)

```python
# --- LLM client setup (do not modify) ---

def get_client():
    """Initialize the LLM client based on available API keys."""
    from openai import OpenAI

    if os.environ.get("OPENROUTER_API_KEY"):
        client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
        return client, "openrouter"

    if os.environ.get("OPENAI_API_KEY"):
        return OpenAI(), "openai"

    raise ValueError(
        "No API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY in .env"
    )


def call_llm(prompt, system_prompt=None, provider=None, client=None):
    """Send a prompt to the LLM and return the response text."""
    if client is None or provider is None:
        client, provider = get_client()

    model = "openai/gpt-4o-mini" if provider == "openrouter" else "gpt-4o-mini"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=1000,
    )
    return response.choices[0].message.content
```

### Load Data

```python
with open("sample_documents/guidelines.txt") as f:
    guidelines_text = f.read()

print(f"Loaded guidelines: {len(guidelines_text)} characters")
print(guidelines_text[:200] + "...")
```

---

## Part 1: PHI Guardrails (no API key needed)

Detect and redact Protected Health Information (PHI) from text using regex patterns. This is a critical safety step before sending clinical text to any LLM.

```python
test_texts = [
    "Patient John Smith, SSN 123-45-6789, presents with chest pain. Contact: 555-867-5309",
    "Email records@hospital.com for MRN#12345. Patient has diabetes.",
    "Call 800-555-0199 or email jane.doe@clinic.org for appointment. BP 140/90.",
    "Blood pressure 120/80, heart rate 72 bpm, SpO2 98%. No acute findings.",
]

print(f"{len(test_texts)} test texts loaded")
for i, t in enumerate(test_texts):
    print(f"  [{i}] {t[:70]}...")
```

### `detect_phi`

Detect common PHI patterns in text using regular expressions.

```python
# TODO: Implement detect_phi
# Requirements:
#   - Check for these PHI patterns using regex:
#     'ssn': SSN format (###-##-####)
#     'phone': Phone numbers (###-###-####, ###.###.####, ##########)
#     'email': Email addresses
#     'mrn': Medical record numbers (MRN or Medical Record followed by digits)
#   - Return a dict mapping PHI type -> list of matches found
#   - Return an empty dict if no PHI is found
#
# Hint: see the lecture's detect_phi() code snippet for the regex patterns
def detect_phi(text):
    pass  # replace with your implementation
```

### `redact_phi`

Replace detected PHI with `[REDACTED]` placeholders.

```python
# TODO: Implement redact_phi
# Requirements:
#   - Takes text and a phi_results dict (from detect_phi)
#   - Replace each matched PHI string in the text with "[REDACTED]"
#   - Return the redacted text
#   - If phi_results is empty, return the original text unchanged
def redact_phi(text, phi_results):
    pass  # replace with your implementation
```

### Test PHI detection and save results (do not modify)

```python
phi_output = []
for i, text in enumerate(test_texts):
    phi_found = detect_phi(text)
    has_phi = len(phi_found) > 0
    redacted = redact_phi(text, phi_found)

    phi_output.append({
        "text_index": i,
        "has_phi": has_phi,
        "phi_found": phi_found,
        "redacted": redacted,
    })

    print(f"Text {i}: PHI={'YES' if has_phi else 'no':3s} | {list(phi_found.keys()) if phi_found else '(clean)'}")
    if has_phi:
        print(f"  Redacted: {redacted[:80]}...")

with open("output/phi_results.json", "w") as f:
    json.dump(phi_output, f, indent=2)

print(f"\nSaved {len(phi_output)} results to output/phi_results.json")
```

---

## Part 2: RAG Pipeline (needs API key)

Build a retrieval-augmented generation pipeline that answers clinical questions grounded in the guidelines document.

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client, provider = get_client()
print(f"Embedding model loaded, LLM provider: {provider}")
```

### `chunk_document`

Split a document into overlapping chunks for retrieval.

```python
# TODO: Implement chunk_document
# Requirements:
#   - Split text into chunks of approximately chunk_size characters
#   - Each chunk should overlap with the next by overlap characters
#   - Split on sentence boundaries ('. ') when possible for cleaner chunks
#   - Return a list of chunk strings
#   - Each chunk should be non-empty
def chunk_document(text, chunk_size=500, overlap=50):
    pass  # replace with your implementation
```

### `retrieve`

Find the most relevant chunks for a query using embedding similarity.

```python
# TODO: Implement retrieve
# Requirements:
#   - Chunk the text using chunk_document()
#   - Embed the query and all chunks using embedding_model.encode()
#   - Compute cosine_similarity between query and chunk embeddings
#   - Return the top n_results chunks as a list of dicts:
#     [{"text": chunk_text, "distance": float}, ...]
#   - Sort by distance ascending (lower = more similar)
#
# Note: cosine_similarity returns similarity (higher = more similar).
# Convert to distance: distance = 1 - similarity
def retrieve(query, text, n_results=3):
    pass  # replace with your implementation
```

### `generate_answer`

Use retrieved chunks as context to generate a grounded answer.

```python
# TODO: Implement generate_answer
# Requirements:
#   - Call retrieve() to get relevant chunks
#   - Build a prompt that includes the retrieved chunks as context
#   - Call call_llm() with a system prompt instructing the model to
#     answer based ONLY on the provided context
#   - Return a dict: {"answer": str, "sources": list of chunk texts, "query": str}
def generate_answer(query, text):
    pass  # replace with your implementation
```

### Test RAG pipeline and save results (do not modify)

```python
test_queries = [
    "What is the first-line treatment for hypertension in a diabetic patient?",
    "How is STEMI diagnosed?",
    "What HbA1c level indicates diabetes?",
]

rag_output = []
for q in test_queries:
    print(f"Q: {q}")
    result = generate_answer(q, guidelines_text)
    print(f"A: {result['answer'][:150]}...")
    print(f"   Sources: {len(result['sources'])} chunks\n")
    rag_output.append(result)

with open("output/rag_results.json", "w") as f:
    json.dump(rag_output, f, indent=2)

print(f"Saved {len(rag_output)} results to output/rag_results.json")
```

---

## Validation

```python
print("Run 'python -m pytest .github/tests/ -v' in your terminal to check your work.")
```

---

## Part 3: Agent Tool Calling *(optional, not graded)*

A working agent with a BMI calculator tool. Experiment with adding tools, changing the system prompt, or giving the agent more complex tasks.

```python
def calculate_bmi(weight_kg: float, height_m: float) -> dict:
    """Calculate BMI from weight and height."""
    bmi = weight_kg / (height_m ** 2)
    category = (
        "Underweight" if bmi < 18.5
        else "Normal weight" if bmi < 25
        else "Overweight" if bmi < 30
        else "Obese"
    )
    return {"bmi": round(bmi, 1), "category": category}


TOOLS = {"calculate_bmi": calculate_bmi}

tool_definitions = [
    {
        "type": "function",
        "function": {
            "name": "calculate_bmi",
            "description": "Calculate Body Mass Index from weight and height",
            "parameters": {
                "type": "object",
                "properties": {
                    "weight_kg": {"type": "number", "description": "Weight in kilograms"},
                    "height_m": {"type": "number", "description": "Height in meters"},
                },
                "required": ["weight_kg", "height_m"],
            },
        },
    }
]


def run_agent(task, max_steps=5):
    """Simple agent loop with tool calling."""
    import json as _json
    messages = [
        {"role": "system", "content": "You are a health assistant. Use tools for calculations."},
        {"role": "user", "content": task},
    ]

    model = "openai/gpt-4o-mini" if provider == "openrouter" else "gpt-4o-mini"

    for step in range(max_steps):
        response = client.chat.completions.create(
            model=model, messages=messages,
            tools=tool_definitions, tool_choice="auto",
        )
        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            return msg.content

        for tc in msg.tool_calls:
            args = _json.loads(tc.function.arguments)
            result = TOOLS[tc.function.name](**args)
            print(f"  Tool: {tc.function.name}({args}) → {result}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": _json.dumps(result)})

    return "Max steps reached"


# Try it
print(run_agent("What's the BMI for someone who is 1.82m and weighs 90kg?"))
```

### Ideas to try

- Add an eGFR calculator tool (see Demo 2 for the formula)
- Add a medication lookup tool
- Give the agent a multi-step task: "Patient weighs 95kg, is 1.70m, and takes metformin. Calculate BMI and check if metformin is appropriate."
- What happens if you ask the agent to do something it has no tool for?
