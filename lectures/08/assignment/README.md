# Assignment 8: LLM Applications — RAG & Guardrails

Build PHI guardrails and a RAG pipeline for clinical guideline Q&A.

**Guidelines data**: `sample_documents/guidelines.txt` — synthetic clinical guidelines covering hypertension, diabetes, and chest pain evaluation.

## Getting Started

```bash
pip install -r requirements.txt
```

### API Key (Part 2 only)

Part 2 requires an LLM API key. We will use [OpenRouter](https://openrouter.ai) (OpenAI-compatible):

1. An API key will be provided on the class forum.
2. (Optional) Sign up for your own free OpenRouter account to create your own API key. There are usually generous free-tier limits for a few models at any given time.
3. Save the API key in `.env` as `OPENROUTER_API_KEY`. For example:

   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```

4. DO NOT COMMIT YOUR API KEY. This will likely invalidate the shared key for everyone and I will have to generate a new one. **This will result in an immediate deduction of one million imaginary points from your final grade**.

Part 1 (PHI guardrails) runs locally and does not need an API key.

## Workflow

Open `assignment.ipynb` and work through both parts:

1. **Part 1: PHI Guardrails** — Implement `detect_phi` and `redact_phi` (no API key needed)
2. **Part 2: RAG Pipeline** — Implement `chunk_document`, `retrieve`, and `generate_answer`
3. **Part 3: Agent Tool Calling** *(optional, not graded)* — Experiment with a working agent

The notebook saves results to `output/` for autograding.

## Output Files

| File | Part | Description |
|:---|:---|:---|
| `output/phi_results.json` | 1 | PHI detection and redaction results for 4 test texts |
| `output/rag_results.json` | 2 | RAG pipeline answers for 3 clinical queries |

## Checking Your Work

```bash
python -m pytest .github/tests/ -v
```

Note: Tests check output artifacts only — run the notebook first, then run tests. You must commit `output/` for CI autograding to pass. Part 3 is optional and has no tests.

## Hints

See `hints.md` for troubleshooting tips.
