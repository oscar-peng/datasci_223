# Demo 3: Agentic Workflow with Tool Calling

In this demo, we'll build an agent that can use tools to complete multi-step tasks.

## Learning Objectives

- Understand how agents differ from single LLM calls
- Implement tool definitions for LLM function calling
- Build an agent loop with tool execution
- Handle multi-step reasoning

## Setup

```python
# %% Setup
# pip install openai python-dotenv

import json
from openai import OpenAI

client = OpenAI()  # Uses OPENAI_API_KEY from environment
```

## Define Tools

```python
# %% Define tools as Python functions

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
    
    return {
        "bmi": round(bmi, 1),
        "category": category
    }

def calculate_egfr(creatinine: float, age: int, is_female: bool, is_black: bool) -> dict:
    """Calculate estimated GFR using CKD-EPI equation."""
    # Simplified CKD-EPI formula
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
    
    if is_black:
        egfr *= 1.159
    
    # Classify CKD stage
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
    
    return {
        "egfr": round(egfr, 1),
        "ckd_stage": stage
    }

def get_medication_info(medication_name: str) -> dict:
    """Look up medication information (simulated database)."""
    medication_db = {
        "metformin": {
            "class": "Biguanide",
            "indication": "Type 2 Diabetes",
            "contraindications": ["eGFR < 30", "Acute kidney injury", "Metabolic acidosis"],
            "common_dose": "500-2000mg daily"
        },
        "lisinopril": {
            "class": "ACE Inhibitor",
            "indication": "Hypertension, Heart failure, Diabetic nephropathy",
            "contraindications": ["Pregnancy", "History of angioedema", "Bilateral renal artery stenosis"],
            "common_dose": "5-40mg daily"
        },
        "amlodipine": {
            "class": "Calcium Channel Blocker",
            "indication": "Hypertension, Angina",
            "contraindications": ["Cardiogenic shock", "Severe aortic stenosis"],
            "common_dose": "5-10mg daily"
        }
    }
    
    med_lower = medication_name.lower()
    if med_lower in medication_db:
        return medication_db[med_lower]
    else:
        return {"error": f"Medication '{medication_name}' not found in database"}

# Tool registry for execution
TOOLS = {
    "calculate_bmi": calculate_bmi,
    "calculate_egfr": calculate_egfr,
    "get_medication_info": get_medication_info
}
```

## Tool Definitions for LLM

```python
# %% Define tools in OpenAI format

tool_definitions = [
    {
        "type": "function",
        "function": {
            "name": "calculate_bmi",
            "description": "Calculate Body Mass Index (BMI) from weight and height",
            "parameters": {
                "type": "object",
                "properties": {
                    "weight_kg": {
                        "type": "number",
                        "description": "Weight in kilograms"
                    },
                    "height_m": {
                        "type": "number", 
                        "description": "Height in meters"
                    }
                },
                "required": ["weight_kg", "height_m"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_egfr",
            "description": "Calculate estimated Glomerular Filtration Rate (eGFR) for kidney function assessment",
            "parameters": {
                "type": "object",
                "properties": {
                    "creatinine": {
                        "type": "number",
                        "description": "Serum creatinine in mg/dL"
                    },
                    "age": {
                        "type": "integer",
                        "description": "Patient age in years"
                    },
                    "is_female": {
                        "type": "boolean",
                        "description": "True if patient is female"
                    },
                    "is_black": {
                        "type": "boolean",
                        "description": "True if patient identifies as Black/African American"
                    }
                },
                "required": ["creatinine", "age", "is_female", "is_black"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_medication_info",
            "description": "Look up information about a medication including class, indications, and contraindications",
            "parameters": {
                "type": "object",
                "properties": {
                    "medication_name": {
                        "type": "string",
                        "description": "Name of the medication to look up"
                    }
                },
                "required": ["medication_name"]
            }
        }
    }
]
```

## Agent Loop Implementation

```python
# %% Agent loop

def execute_tool(tool_call):
    """Execute a tool call and return the result."""
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    
    if function_name in TOOLS:
        result = TOOLS[function_name](**arguments)
        return json.dumps(result)
    else:
        return json.dumps({"error": f"Unknown tool: {function_name}"})

def run_agent(task: str, max_steps: int = 5, verbose: bool = True) -> str:
    """
    Run an agent that can use tools to complete a task.
    
    Parameters
    ----------
    task : str
        The task to complete
    max_steps : int
        Maximum number of tool-calling iterations
    verbose : bool
        Whether to print intermediate steps
    
    Returns
    -------
    str
        The agent's final response
    """
    messages = [
        {
            "role": "system",
            "content": """You are a helpful clinical assistant. You have access to tools for:
- Calculating BMI
- Calculating eGFR (kidney function)
- Looking up medication information

Use these tools when needed to provide accurate clinical assessments.
Always show your reasoning and calculations."""
        },
        {"role": "user", "content": task}
    ]
    
    for step in range(max_steps):
        if verbose:
            print(f"\n--- Step {step + 1} ---")
        
        # Call LLM
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tool_definitions,
            tool_choice="auto"
        )
        
        assistant_message = response.choices[0].message
        messages.append(assistant_message)
        
        # Check if we're done (no tool calls)
        if not assistant_message.tool_calls:
            if verbose:
                print("Agent completed task (no more tool calls)")
            return assistant_message.content
        
        # Execute tool calls
        for tool_call in assistant_message.tool_calls:
            if verbose:
                print(f"Calling tool: {tool_call.function.name}")
                print(f"Arguments: {tool_call.function.arguments}")
            
            result = execute_tool(tool_call)
            
            if verbose:
                print(f"Result: {result}")
            
            # Add tool result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })
    
    if verbose:
        print(f"\nMax steps ({max_steps}) reached")
    return "Task incomplete - max steps reached"
```

## Test Agent with Clinical Tasks

```python
# %% Test with various clinical tasks

# Task 1: Simple calculation
print("=" * 60)
print("TASK 1: BMI Calculation")
print("=" * 60)
task1 = "Calculate the BMI for a patient who weighs 85 kg and is 1.75 meters tall."
result1 = run_agent(task1)
print(f"\nFinal Response:\n{result1}")

# Task 2: Multi-step reasoning
print("\n" + "=" * 60)
print("TASK 2: Medication Safety Check")
print("=" * 60)
task2 = """A 68-year-old female patient has a creatinine of 1.8 mg/dL. 
Can she safely take metformin? Calculate her eGFR first and then check the medication contraindications."""
result2 = run_agent(task2)
print(f"\nFinal Response:\n{result2}")

# Task 3: Complex clinical scenario
print("\n" + "=" * 60)
print("TASK 3: Patient Assessment")
print("=" * 60)
task3 = """New patient:
- 55-year-old male
- Weight: 95 kg, Height: 1.80 m
- Creatinine: 1.2 mg/dL
- Current medications: lisinopril, amlodipine

Please:
1. Calculate his BMI and categorize it
2. Calculate his eGFR to assess kidney function
3. Look up his current medications and note any concerns"""
result3 = run_agent(task3)
print(f"\nFinal Response:\n{result3}")
```

## Bonus: Using MCP for Standardized Tool Access

The tools we built manually above work great, but in production you'd want a standardized approach. **Model Context Protocol (MCP)** provides exactly that—plug-and-play servers that expose tools in a consistent format.

### Installing MCP

```python
# %% Install MCP (run once)
# pip install mcp
```

### Discovering Tools from an MCP Server

```python
# %% MCP tool discovery
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def explore_mcp_server():
    """Connect to an MCP server and list available tools."""
    # The filesystem server is a good example - it exposes read/write tools
    server = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "."]
    )
    
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Discover available tools
            tools = await session.list_tools()
            print("Available MCP Tools:")
            print("-" * 40)
            for tool in tools.tools:
                print(f"  {tool.name}: {tool.description[:60]}...")
            
            return tools

# Uncomment to run (requires Node.js/npx installed):
# asyncio.run(explore_mcp_server())
```

### Converting MCP Tools to OpenAI Format

```python
# %% MCP to OpenAI format converter

def mcp_to_openai_tools(mcp_tools):
    """Convert MCP tool definitions to OpenAI function calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        }
        for tool in mcp_tools.tools
    ]

# Once converted, these tools work with our existing agent loop!
# openai_tools = mcp_to_openai_tools(mcp_tools)
```

### Why MCP Matters

| Manual Approach (this demo) | MCP Approach |
|:---|:---|
| Define each tool function yourself | Use pre-built servers |
| Wire up tool execution manually | Standard call/response protocol |
| Build each integration from scratch | Plug-and-play servers |
| Great for learning | Great for production |

**Key insight**: The agent loop stays the same—MCP just standardizes how tools are discovered and called. What we learned about tool definitions and agent loops applies directly to MCP.

## Exercises

1. **Add new tools**: Implement a drug interaction checker or dosing calculator
2. **Add guardrails**: Implement validation to prevent incorrect tool arguments
3. **Add memory**: Allow the agent to reference previous calculations in a session
4. **Error handling**: Make the agent more robust to tool failures
5. **Try MCP**: Install an MCP server and connect your agent to it

## Key Takeaways

- Agents can autonomously decide when and how to use tools
- Tool definitions specify the interface for the LLM
- The agent loop iterates until the task is complete
- Proper tool design is crucial for agent effectiveness
- Always validate tool inputs and handle errors gracefully
- **MCP** standardizes tool integration for production use
