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

# Demo 3: LLM API & Prompt Engineering

Calling an LLM via API: send a prompt, get structured results back. The techniques here apply directly to the assignment's `extractor.py`.

## Setup

```python
%pip install -q openai python-dotenv matplotlib numpy
```

```python
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# OpenRouter is an aggregator — same openai SDK, different base_url
if os.environ.get("OPENROUTER_API_KEY"):
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    MODEL = "openai/gpt-4o-mini"
    print("Using OpenRouter")
elif os.environ.get("OPENAI_API_KEY"):
    client = OpenAI()
    MODEL = "gpt-4o-mini"
    print("Using OpenAI directly")
else:
    raise ValueError("Set OPENROUTER_API_KEY or OPENAI_API_KEY in a .env file")
```

We'll use the same clinical note throughout so we can compare techniques directly. This is Note 1 from the assignment's `clinical_notes.txt`.

```python
note = """Patient is a 58-year-old male presenting with chest pain radiating to the left arm.
Blood pressure 145/92 mmHg, heart rate 88 bpm. Troponin elevated at 0.8 ng/mL.
ECG shows ST elevation in leads V1-V4. Patient started on aspirin 325mg and
heparin drip. Diagnosis: Acute ST-elevation myocardial infarction (STEMI)."""

print(note)
```

A reusable helper for calling the LLM. This matches the `call_llm` function already provided in the assignment's `extractor.py`.

```python
def call_llm(prompt, system="You are a medical information extraction assistant.", temperature=0):
    """Send a prompt to the LLM and return the response text."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=500,
    )
    return response.choices[0].message.content
```

## Zero-Shot Prompting

Zero-shot: describe the task with no examples. Start here; add complexity only if needed.

```python
zero_shot_prompt = f"""Extract the primary diagnosis from this clinical note.

Clinical Note:
{note}

Diagnosis:"""

response = call_llm(zero_shot_prompt)
print(response)
```

The model returns free text — a natural language string. That's fine for a human reader, but what if you need to store this in a database, validate it against a codebook, or feed it into another program? You'd have to parse the text with regex or string matching, which is fragile and error-prone.

This is the core tension of this demo: **unstructured output** (free text) vs **structured output** (JSON). We'll try several prompting techniques and compare them at the end.

## Few-Shot Prompting

Few-shot provides examples before the actual input. The model infers the pattern from examples rather than instructions alone.

```python
one_shot_prompt = f"""Extract the primary diagnosis from clinical notes.

Example:
Note: "65-year-old female with polyuria, polydipsia, fasting glucose 285 mg/dL, HbA1c 9.2%."
Diagnosis: Type 2 Diabetes Mellitus (poorly controlled)

Now extract the diagnosis:
Note: "{note}"
Diagnosis:"""

response = call_llm(one_shot_prompt)
print(response)
```

```python
two_shot_prompt = f"""Extract the primary diagnosis from clinical notes.

Example 1:
Note: "65-year-old female with polyuria, polydipsia, fasting glucose 285 mg/dL, HbA1c 9.2%."
Diagnosis: Type 2 Diabetes Mellitus (poorly controlled)

Example 2:
Note: "42-year-old male with productive cough, fever to 101.5F, right lower lobe infiltrate on X-ray."
Diagnosis: Community-acquired pneumonia

Now extract the diagnosis:
Note: "{note}"
Diagnosis:"""

response = call_llm(two_shot_prompt)
print(response)
```

Few-shot tends to produce more consistent formatting because the model mirrors the style of the examples.

## Structured Outputs (JSON Extraction)

Free-text responses are hard to use downstream. If you want to write results to a database, pass them to another function, or validate them, you need a **schema** — a contract specifying field names, types, and structure.

The standard approach: put the schema directly in the prompt and tell the model to return _only_ JSON. This is the single most practical technique in this demo — it turns an LLM from a text generator into a data extraction tool.

Note the double braces `{{` / `}}` — Python f-strings use single braces for variable interpolation, so literal braces in JSON must be doubled.

```python
schema_prompt = f"""Extract structured information from this clinical note.
Return ONLY a JSON object with exactly these fields:

{{
  "diagnosis": "<primary diagnosis as a string>",
  "medications": ["<list of medications mentioned>"],
  "lab_values": {{"<test name>": "<value with units>"}},
  "confidence": <float 0.0 to 1.0>
}}

Clinical Note:
{note}"""

response = call_llm(schema_prompt)
print(response)
```

The response is usually valid JSON, but LLMs sometimes wrap it in markdown code fences or add commentary. We need a parser that handles all three cases.

```python
def parse_json_response(text):
    """Extract JSON from an LLM response, handling markdown code fences."""
    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip markdown code fences
    if "```" in text:
        start = text.find("```")
        end = text.rfind("```")
        if start != end:
            block = text[start:end]
            lines = block.split("\n")
            # Drop the opening ``` line (may include language label like ```json)
            block = "\n".join(lines[1:])
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                pass

    # Strategy 3: find outermost { ... }
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    return None


parsed = parse_json_response(response)
if parsed:
    print(json.dumps(parsed, indent=2))
else:
    print("Could not parse JSON. Raw response:")
    print(response)
```

Now validate that the parsed result has the fields we asked for.

```python
def validate_response(response):
    """Check that the response has all required fields."""
    if not isinstance(response, dict):
        return False
    required = {"diagnosis", "medications", "lab_values", "confidence"}
    return required.issubset(response.keys())


if parsed:
    print(f"Valid: {validate_response(parsed)}")
    print(f"Diagnosis:   {parsed.get('diagnosis')}")
    print(f"Medications: {parsed.get('medications')}")
    print(f"Lab values:  {parsed.get('lab_values')}")
    print(f"Confidence:  {parsed.get('confidence')}")
```

## Few-Shot + JSON

When zero-shot JSON extraction produces inconsistent schemas (wrong field names, missing fields), adding a complete JSON example anchors the format.

```python
few_shot_json_prompt = f"""Extract structured information from clinical notes. Return JSON only.

Example:
Note: "65-year-old female with polyuria, polydipsia. Fasting glucose 285 mg/dL, HbA1c 9.2%.
Taking metformin 1000mg BID and lisinopril 10mg daily."

Output:
{{
  "diagnosis": "Type 2 Diabetes Mellitus (poorly controlled)",
  "medications": ["metformin 1000mg BID", "lisinopril 10mg daily"],
  "lab_values": {{"fasting_glucose": "285 mg/dL", "HbA1c": "9.2%"}},
  "confidence": 0.95
}}

Now extract from this note:
Note: "{note}"

Output:"""

response = call_llm(few_shot_json_prompt)
parsed_few = parse_json_response(response)
print(json.dumps(parsed_few, indent=2) if parsed_few else response)
```

The example anchors field names, value formats (units in lab values, full dosing in medications). This is the pattern the assignment's `build_prompt(few_shot=True)` should implement.

## Chain-of-Thought Prompting

Chain-of-thought (CoT) asks the model to reason step by step before producing the answer. By "thinking out loud," the model is less likely to skip details or make errors on complex cases. The tradeoff: more output tokens (slower, costs more), but often higher accuracy and — critically — **visible reasoning** you can audit.

For clinical extraction, CoT is especially useful when the note is ambiguous (e.g., multiple possible diagnoses, unclear medication dosing).

```python
cot_prompt = f"""Extract structured data from the clinical note below.

First, reason through the key elements step by step:
1. What is the primary diagnosis?
2. What medications are mentioned (include dose and frequency if given)?
3. What lab values are reported (include units)?
4. How confident are you in this extraction (0.0-1.0)?

Then produce the final JSON:
{{
  "diagnosis": "<primary diagnosis>",
  "medications": ["<list>"],
  "lab_values": {{"<name>": "<value>"}},
  "confidence": <float>
}}

Clinical Note:
{note}"""

response = call_llm(cot_prompt)
print(response)
```

```python
parsed_cot = parse_json_response(response)
if parsed_cot:
    print("\nExtracted JSON:")
    print(json.dumps(parsed_cot, indent=2))
```

The reasoning is visible (useful for debugging) and the same `parse_json_response` function works — it finds the JSON block at the end regardless of the preceding text.

## Comparing Approaches

We've now seen five prompting techniques on the same clinical note. Let's collect them all and compare: which produced structured output? Which got the right answer? Which extracted the most detail?

```python
# Ground truth for the STEMI note
ground_truth = {
    "diagnosis": "Acute ST-elevation myocardial infarction (STEMI)",
    "medications": ["aspirin 325mg", "heparin drip"],
    "lab_values": {"troponin": "0.8 ng/mL"},
}

# Collect all results
all_results = {}

# Re-run all techniques and store results
# Zero-shot (unstructured)
zero_response = call_llm(zero_shot_prompt)
all_results["Zero-shot"] = {"raw": zero_response, "structured": False, "parsed": None}

# One-shot (unstructured)
one_response = call_llm(one_shot_prompt)
all_results["One-shot"] = {"raw": one_response, "structured": False, "parsed": None}

# Two-shot (unstructured)
two_response = call_llm(two_shot_prompt)
all_results["Two-shot"] = {"raw": two_response, "structured": False, "parsed": None}

# Schema JSON (structured)
schema_response = call_llm(schema_prompt)
schema_parsed = parse_json_response(schema_response)
all_results["Schema JSON"] = {"raw": schema_response, "structured": True, "parsed": schema_parsed}

# Few-shot JSON (structured)
fewshot_response = call_llm(few_shot_json_prompt)
fewshot_parsed = parse_json_response(fewshot_response)
all_results["Few-shot JSON"] = {"raw": fewshot_response, "structured": True, "parsed": fewshot_parsed}

# Chain-of-thought (structured)
cot_response = call_llm(cot_prompt)
cot_parsed = parse_json_response(cot_response)
all_results["Chain-of-thought"] = {"raw": cot_response, "structured": True, "parsed": cot_parsed}

print(f"Collected {len(all_results)} results")
```

### Correctness Evaluation

For structured outputs, we can programmatically check whether the extracted fields match the ground truth. This is the advantage — structured outputs are machine-checkable.

```python
def evaluate_extraction(parsed, ground_truth):
    """Score an extraction against ground truth. Returns a dict of field scores."""
    if not parsed:
        return {"diagnosis": False, "medications": False, "lab_values": False, "valid_json": False}

    scores = {"valid_json": True}

    # Diagnosis: check if ground truth diagnosis appears in extracted diagnosis
    diag = parsed.get("diagnosis", "")
    scores["diagnosis"] = "stemi" in diag.lower() or "st-elevation" in diag.lower()

    # Medications: check each ground truth med appears somewhere in the extracted list
    extracted_meds = [m.lower() for m in parsed.get("medications", [])]
    meds_text = " ".join(extracted_meds)
    scores["medications"] = "aspirin" in meds_text and "heparin" in meds_text

    # Lab values: check troponin is present
    labs = parsed.get("lab_values", {})
    labs_text = json.dumps(labs).lower()
    scores["lab_values"] = "troponin" in labs_text or "0.8" in labs_text

    return scores


# Evaluate structured methods
print(f"{'Method':<20} {'Valid JSON':<12} {'Diagnosis':<12} {'Medications':<14} {'Lab Values':<12}")
print("-" * 70)

for name, result in all_results.items():
    if result["structured"]:
        scores = evaluate_extraction(result["parsed"], ground_truth)
        row = f"{name:<20}"
        for field in ["valid_json", "diagnosis", "medications", "lab_values"]:
            check = "Y" if scores[field] else "N"
            row += f" {check:<13}"
        print(row)
    else:
        print(f"{name:<20} {'(free text)':<12} {'--':<12} {'--':<14} {'--':<12}")
```

### Side-by-Side Comparison

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: structured vs unstructured output type
methods = list(all_results.keys())
is_structured = [all_results[m]["structured"] for m in methods]
colors = ['#2ecc71' if s else '#e74c3c' for s in is_structured]

axes[0].barh(methods, [1] * len(methods), color=colors)
axes[0].set_xlim(0, 1.2)
axes[0].set_xticks([])
axes[0].set_title('Output Type')
for i, (m, s) in enumerate(zip(methods, is_structured)):
    axes[0].text(0.5, i, 'Structured (JSON)' if s else 'Unstructured (text)',
                 ha='center', va='center', fontweight='bold', color='white')

# Right: correctness scores for structured methods
structured_methods = [m for m in methods if all_results[m]["structured"]]
fields = ["diagnosis", "medications", "lab_values"]
field_labels = ["Diagnosis", "Medications", "Lab Values"]

score_matrix = []
for m in structured_methods:
    scores = evaluate_extraction(all_results[m]["parsed"], ground_truth)
    score_matrix.append([scores[f] for f in fields])

score_array = np.array(score_matrix, dtype=float)
im = axes[1].imshow(score_array, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
axes[1].set_xticks(range(len(field_labels)))
axes[1].set_xticklabels(field_labels)
axes[1].set_yticks(range(len(structured_methods)))
axes[1].set_yticklabels(structured_methods)
axes[1].set_title('Extraction Correctness')

for i in range(len(structured_methods)):
    for j in range(len(fields)):
        axes[1].text(j, i, 'Y' if score_array[i, j] else 'N',
                     ha='center', va='center', fontweight='bold', fontsize=14)

plt.tight_layout()
plt.show()
```

The key takeaway: all three structured methods extract the correct information, but they differ in what _extra_ detail they provide. Few-shot JSON tends to mirror the example's field structure, while Chain-of-thought produces explicit reasoning you can audit. For the assignment, any of the three structured approaches will work — pick the one that feels clearest.

## Batch Extraction (Assignment Preview)

The assignment asks you to run extraction on all four notes in `clinical_notes.txt`. Here's the end-to-end workflow.

```python
def load_notes(filepath):
    """Load notes from the assignment's clinical_notes.txt format."""
    with open(filepath) as f:
        content = f.read()
    sections = content.split("## Note")
    return [s.split("\n", 1)[1].strip() for s in sections[1:] if s.strip()]


# Find the notes file (works from demo/ or repo root)
candidates = [
    "../assignment/clinical_notes.txt",
    "lectures/07/assignment/clinical_notes.txt",
]
notes_path = next((p for p in candidates if os.path.exists(p)), None)

if notes_path:
    notes = load_notes(notes_path)
    print(f"Loaded {len(notes)} notes")
    for i, n in enumerate(notes, 1):
        print(f"\n--- Note {i} ---\n{n[:80]}...")
else:
    print("clinical_notes.txt not found. Run from lectures/07/demo/ or repo root.")
```

```python
if notes_path:
    results = []
    for i, note_text in enumerate(notes, 1):
        print(f"\nProcessing Note {i}...")
        prompt = f"""Extract structured information from clinical notes. Return JSON only.

Example:
Note: "65-year-old female with polyuria, polydipsia. Fasting glucose 285 mg/dL, HbA1c 9.2%.
Taking metformin 1000mg BID and lisinopril 10mg daily."

Output:
{{
  "diagnosis": "Type 2 Diabetes Mellitus (poorly controlled)",
  "medications": ["metformin 1000mg BID", "lisinopril 10mg daily"],
  "lab_values": {{"fasting_glucose": "285 mg/dL", "HbA1c": "9.2%"}},
  "confidence": 0.95
}}

Now extract from this note:
Note: "{note_text}"

Output:"""
        response = call_llm(prompt)
        parsed = parse_json_response(response)
        if parsed and validate_response(parsed):
            results.append({"note_id": i, **parsed})
            print(f"  Diagnosis: {parsed['diagnosis']}")
            print(f"  Confidence: {parsed['confidence']}")
        else:
            print(f"  Extraction failed. Raw: {response[:100]}")

    print(f"\n{len(results)}/{len(notes)} notes extracted successfully")
```

