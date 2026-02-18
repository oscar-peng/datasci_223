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

# Demo 2: Workflow Patterns & Agents

Build up from simple prompt chaining to a full agent with tools, then see how the OpenAI Agents SDK packages the same ideas.

## Learning Objectives

- Implement prompt chaining for multi-step clinical text processing
- Add PHI guardrails to LLM calls
- Build an agent loop with tool calling
- Use the OpenAI Agents SDK for a cleaner agent implementation

## Setup

```python
%pip install -q openai openai-agents python-dotenv
```

```python
import os
import re
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

if os.environ.get("OPENROUTER_API_KEY"):
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    MODEL = "openai/gpt-4o-mini"
elif os.environ.get("OPENAI_API_KEY"):
    client = OpenAI()
    MODEL = "gpt-4o-mini"
else:
    raise ValueError("Set OPENROUTER_API_KEY or OPENAI_API_KEY in .env")


def llm_call(prompt: str) -> str:
    """Simple wrapper for chat completion."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


print(f"Using model: {MODEL}")
```

## Section 1: Prompt Chaining

Each step in a chain is simpler, more testable, and produces an intermediate artifact you can inspect. If step 2 fails, you know exactly where.

```python
clinical_note = """
Patient is a 72-year-old male presenting with increasing shortness of breath
over the past 3 days. History of COPD, type 2 diabetes on metformin 1000mg BID,
and hypertension on lisinopril 20mg daily. Vitals: BP 158/92, HR 96, SpO2 89%
on room air. Chest X-ray shows bilateral infiltrates. Started on supplemental
oxygen, azithromycin 500mg, and ceftriaxone 1g IV. Labs: WBC 14.2, glucose 245,
creatinine 1.4, BNP 890.
"""

# Step 1: Extract entities
entities = llm_call(
    f"Extract all medical entities from this note as a bulleted list:\n{clinical_note}"
)
print("STEP 1 — Entities:")
print(entities)
```

```python
# Step 2: Classify entities by type
classified = llm_call(
    f"Classify these entities by type (condition, medication, lab value, vital sign):\n{entities}"
)
print("STEP 2 — Classified:")
print(classified)
```

```python
# Step 3: Summarize
summary = llm_call(
    f"Write a brief clinical summary (3-4 sentences) based on:\n{classified}"
)
print("STEP 3 — Summary:")
print(summary)
```

Each step used a different prompt, and we could inspect the intermediate outputs. Compare this to stuffing everything into one giant prompt — harder to debug, harder to test.

## Section 2: Guardrails

Before sending text to an LLM, check for Protected Health Information. This uses simple regex patterns — production systems would use NLP models like Presidio.

```python
def detect_phi(text: str) -> dict | None:
    """Detect common PHI patterns via regex."""
    patterns = {
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "mrn": r"\b(MRN|Medical Record)[\s:#]*\d+\b",
    }

    found = {}
    for phi_type, pattern in patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            found[phi_type] = matches

    return found if found else None


def safe_llm_call(prompt: str) -> str:
    """LLM call with input/output PHI guardrails."""
    phi_in = detect_phi(prompt)
    if phi_in:
        raise ValueError(f"PHI detected in input: {list(phi_in.keys())}")

    result = llm_call(prompt)

    phi_out = detect_phi(result)
    if phi_out:
        raise ValueError(f"PHI detected in output: {list(phi_out.keys())}")

    return result
```

```python
# Safe text — passes guardrails
clean_result = safe_llm_call("Summarize the treatment for stage 1 hypertension.")
print("Clean input result:")
print(clean_result[:200] + "...")
```

```python
# Dangerous text — blocked by guardrails
try:
    safe_llm_call(
        "Summarize this note: Patient John Smith, SSN 123-45-6789, MRN#12345, "
        "presents with chest pain. Contact: 555-867-5309, john@hospital.com"
    )
except ValueError as e:
    print(f"BLOCKED: {e}")
```

## Section 3: Tool Definitions & Agent Loop

Agents decide *when* to use tools and *what arguments* to pass. We define tools as Python functions, describe them in JSON for the LLM, then loop until the model stops requesting tool calls.

```python
def calculate_bmi(weight_kg: float, height_m: float) -> dict:
    """Calculate BMI from weight and height."""
    bmi = weight_kg / (height_m ** 2)
    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal weight"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"
    return {"bmi": round(bmi, 1), "category": category}


def calculate_egfr(creatinine: float, age: int, is_female: bool) -> dict:
    """Calculate estimated GFR using simplified CKD-EPI equation."""
    if is_female:
        if creatinine <= 0.7:
            egfr = 144 * ((creatinine / 0.7) ** -0.329) * (0.993 ** age)
        else:
            egfr = 144 * ((creatinine / 0.7) ** -1.209) * (0.993 ** age)
    else:
        if creatinine <= 0.9:
            egfr = 141 * ((creatinine / 0.9) ** -0.411) * (0.993 ** age)
        else:
            egfr = 141 * ((creatinine / 0.9) ** -1.209) * (0.993 ** age)

    if egfr >= 90:
        stage = "G1 (Normal)"
    elif egfr >= 60:
        stage = "G2 (Mild)"
    elif egfr >= 45:
        stage = "G3a (Mild-Moderate)"
    elif egfr >= 30:
        stage = "G3b (Moderate-Severe)"
    elif egfr >= 15:
        stage = "G4 (Severe)"
    else:
        stage = "G5 (Kidney Failure)"

    return {"egfr": round(egfr, 1), "ckd_stage": stage}


def get_medication_info(medication_name: str) -> dict:
    """Look up medication information (simulated database)."""
    db = {
        "metformin": {
            "class": "Biguanide",
            "indication": "Type 2 Diabetes",
            "contraindications": ["eGFR < 30", "Acute kidney injury", "Metabolic acidosis"],
            "common_dose": "500-2000mg daily",
        },
        "lisinopril": {
            "class": "ACE Inhibitor",
            "indication": "Hypertension, Heart failure, Diabetic nephropathy",
            "contraindications": ["Pregnancy", "History of angioedema"],
            "common_dose": "5-40mg daily",
        },
        "amlodipine": {
            "class": "Calcium Channel Blocker",
            "indication": "Hypertension, Angina",
            "contraindications": ["Cardiogenic shock", "Severe aortic stenosis"],
            "common_dose": "5-10mg daily",
        },
    }
    med = medication_name.lower()
    return db.get(med, {"error": f"'{medication_name}' not found in database"})


TOOLS = {
    "calculate_bmi": calculate_bmi,
    "calculate_egfr": calculate_egfr,
    "get_medication_info": get_medication_info,
}
```

```python
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
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_egfr",
            "description": "Calculate estimated GFR for kidney function assessment",
            "parameters": {
                "type": "object",
                "properties": {
                    "creatinine": {"type": "number", "description": "Serum creatinine in mg/dL"},
                    "age": {"type": "integer", "description": "Patient age in years"},
                    "is_female": {"type": "boolean", "description": "True if patient is female"},
                },
                "required": ["creatinine", "age", "is_female"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_medication_info",
            "description": "Look up medication class, indications, and contraindications",
            "parameters": {
                "type": "object",
                "properties": {
                    "medication_name": {"type": "string", "description": "Medication name"},
                },
                "required": ["medication_name"],
            },
        },
    },
]
```

```python
def execute_tool(tool_call):
    """Execute a tool call and return the result as JSON string."""
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    if name in TOOLS:
        return json.dumps(TOOLS[name](**args))
    return json.dumps({"error": f"Unknown tool: {name}"})


def run_agent(task: str, max_steps: int = 5, verbose: bool = True) -> str:
    """Run an agent that uses tools to complete a clinical task."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a clinical assistant with tools for BMI calculation, "
                "eGFR calculation, and medication lookup. Use tools for calculations "
                "— never do math in your head. Show your reasoning."
            ),
        },
        {"role": "user", "content": task},
    ]

    for step in range(max_steps):
        if verbose:
            print(f"\n--- Step {step + 1} ---")

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tool_definitions,
            tool_choice="auto",
        )

        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            if verbose:
                print("Agent completed (no more tool calls)")
            return msg.content

        for tc in msg.tool_calls:
            if verbose:
                print(f"Tool: {tc.function.name}({tc.function.arguments})")
            result = execute_tool(tc)
            if verbose:
                print(f"Result: {result}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return "Max steps reached"
```

```python
print("=" * 60)
print("TASK: Simple BMI calculation")
print("=" * 60)
result = run_agent("Calculate BMI for a patient who weighs 85 kg and is 1.75m tall.")
print(f"\nFinal:\n{result}")
```

```python
print("=" * 60)
print("TASK: Multi-step medication safety check")
print("=" * 60)
result = run_agent(
    "A 68-year-old female has creatinine 1.8 mg/dL. "
    "Can she safely take metformin? Check her kidney function first."
)
print(f"\nFinal:\n{result}")
```

## Section 4: OpenAI Agents SDK

The Agents SDK packages the same pattern — tools, agent loop, message handling — into a cleaner API. Same concepts, less boilerplate.

```python
from openai import AsyncOpenAI
from agents import Agent, Runner, function_tool, set_default_openai_client

# Configure the Agents SDK to use the same OpenRouter credentials
if os.environ.get("OPENROUTER_API_KEY"):
    set_default_openai_client(AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    ))
    AGENTS_MODEL = "openai/gpt-4o-mini"
else:
    AGENTS_MODEL = "gpt-4o-mini"


@function_tool
def calculate_bmi_sdk(weight_kg: float, height_m: float) -> str:
    """Calculate BMI from weight and height."""
    bmi = weight_kg / (height_m ** 2)
    category = (
        "Underweight" if bmi < 18.5
        else "Normal weight" if bmi < 25
        else "Overweight" if bmi < 30
        else "Obese"
    )
    return f"BMI: {bmi:.1f} ({category})"


agent = Agent(
    name="Health Assistant",
    model=AGENTS_MODEL,
    instructions="You help with health data analysis. Use tools for all calculations.",
    tools=[calculate_bmi_sdk],
)

# Jupyter supports top-level await — use async Runner.run() instead of run_sync()
result = await Runner.run(agent, "Calculate BMI for a 75kg patient who is 1.75m tall")
print(result.final_output)
```

Compare the two approaches:

| Manual Agent (Section 3) | Agents SDK (Section 4) |
|:---|:---|
| Write tool JSON schema by hand | `@function_tool` decorator |
| Implement `run_agent()` loop | `await Runner.run()` |
| Manage message history yourself | Handled automatically |
| Full control over every step | Cleaner API, less boilerplate |
| Great for learning | Great for production |

## Exercises

1. **Add a tool**: Add a drug interaction checker to the manual agent (e.g., "does metformin interact with lisinopril?")
2. **Chain + guardrails**: Wrap the prompt chain from Section 1 with the PHI guardrails from Section 2
3. **Agents SDK tools**: Add `calculate_egfr` and `get_medication_info` to the SDK agent
4. **Error handling**: What happens when the agent gets bad tool arguments? Add validation.

## Key Takeaways

- Prompt chaining breaks complex tasks into simple, testable steps
- Guardrails enforce safety rules on LLM inputs and outputs
- Tool-calling agents autonomously decide when and how to use functions
- The Agents SDK packages these patterns with less boilerplate
