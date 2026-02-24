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

# Demo 1: Building a Clinical Agent

Define tools, wire up function calling, and build an agent loop that autonomously decides which tools to call and in what order.

## Learning Objectives

- Define clinical tools as Python functions with JSON schemas
- Make a single function-calling API request and walk through the 3-step dance
- Build a multi-tool agent loop that decides when and which tools to call
- See the same agent built with the OpenAI Agents SDK

## Setup

```python
%pip install -q openai openai-agents python-dotenv
```

```python
import os
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

print(f"Using model: {MODEL}")
```

## Section 1: Function Calling Basics

The lecture showed the concept — now let's see the full round-trip. Function calling is a 3-step dance:

1. **You send** the request with tool definitions
2. **The model responds** with a tool call (name + arguments) instead of text
3. **You execute** the function and feed the result back

```python
# A real clinical tool: BMI calculation with category
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


# Describe the tool in JSON schema — this is what the model sees
bmi_tool = {
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
```

```python
# Step 1: Send request with tools
response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "What's the BMI for a 85kg, 1.75m patient?"}],
    tools=[bmi_tool],
    tool_choice="auto",
)

msg = response.choices[0].message
print("Model wants to call a tool:")
print(f"  Function: {msg.tool_calls[0].function.name}")
print(f"  Arguments: {msg.tool_calls[0].function.arguments}")
```

```python
# Step 2: Execute the tool ourselves
tool_call = msg.tool_calls[0]
args = json.loads(tool_call.function.arguments)
result = calculate_bmi(**args)
print(f"Tool result: {result}")
```

```python
# Step 3: Feed the result back and get the final answer
messages = [
    {"role": "user", "content": "What's the BMI for a 85kg, 1.75m patient?"},
    msg,  # the model's tool call message
    {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)},
]

final = client.chat.completions.create(model=MODEL, messages=messages)
print(final.choices[0].message.content)
```

That's function calling. The model doesn't run the function — it tells you *which* function to call and *what arguments* to pass. You execute it and return the result. This separation is what makes agents safe: your code controls what actually happens.

## Section 2: Multi-Tool Agent

One tool is useful. Multiple tools with a loop is an **agent** — the model decides which tools to call and in what order, iterating until the task is complete.

We'll define three clinical tools: BMI calculation, estimated GFR (kidney function), and a medication lookup database. Each is a plain Python function that returns a dict.

```python
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

print(f"Defined {len(TOOLS)} tools: {', '.join(TOOLS.keys())}")
```

The model can't call Python functions directly — it needs JSON schemas that describe each tool's name, purpose, and parameter types. This is the contract between your code and the model.

```python
tool_definitions = [
    bmi_tool,
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

print(f"Registered {len(tool_definitions)} tool schemas for the model")
for td in tool_definitions:
    print(f"  {td['function']['name']}: {td['function']['description']}")
```

Now we wire it together. `execute_tool` dispatches a tool call to the right Python function. `run_agent` loops: send the conversation to the model → if it requests tool calls, execute them and feed results back → repeat until the model responds with text instead of tool calls.

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

Watch the steps: the agent calls `calculate_egfr` first (to check kidney function), then `get_medication_info` (to find contraindications), then reasons about whether metformin is safe. It planned the sequence itself.

## Section 3: OpenAI Agents SDK

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

| Manual Agent (Section 2) | Agents SDK (Section 3) |
|:---|:---|
| Write tool JSON schema by hand | `@function_tool` decorator |
| Implement `run_agent()` loop | `await Runner.run()` |
| Manage message history yourself | Handled automatically |
| Full control over every step | Cleaner API, less boilerplate |
| Great for learning | Great for production |

## Exercises

1. **Add a tool**: Add a drug interaction checker to the manual agent (e.g., "does metformin interact with lisinopril?")
2. **Agents SDK tools**: Add `calculate_egfr` and `get_medication_info` to the SDK agent
3. **Error handling**: What happens when the agent gets bad tool arguments? Add validation to `execute_tool`

## Key Takeaways

- Function calling is a 3-step dance: send tools → model requests tool → execute and return
- Tool-calling agents autonomously decide when and how to use functions
- The agent loop (plan → act → observe → repeat) is the core pattern behind Claude Code, ChatGPT, etc.
- The Agents SDK packages these patterns with less boilerplate
