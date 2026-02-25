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

# Assignment 8: Murder Mystery Agents

Build detective agents that solve two murder mysteries using the OpenAI Agents SDK.

## Setup

```python
%pip install -q openai>=1.0.0 openai-agents>=0.1.0 python-dotenv>=1.0.0 pydantic>=2.0.0
```

```python
%reset -f

import os
import json
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
from agents import Agent, Runner, function_tool, set_default_openai_client

load_dotenv()
os.makedirs("output", exist_ok=True)

# Configure API client
if os.environ.get("OPENROUTER_API_KEY"):
    set_default_openai_client(AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    ))
    AGENTS_MODEL = "openai/gpt-4o-mini"
elif os.environ.get("OPENAI_API_KEY"):
    AGENTS_MODEL = "gpt-4o-mini"
else:
    raise ValueError("Set OPENROUTER_API_KEY or OPENAI_API_KEY in .env")

print(f"Using model: {AGENTS_MODEL}")
```

## Transcript Logger

A utility to capture what the detective agent does — which tools it calls and what it learns.

```python
transcript = []

def log_action(action_type: str, detail: str):
    """Append an entry to the investigation transcript."""
    transcript.append({"action": action_type, "detail": detail})
    print(f"  [{action_type}] {detail[:120]}")
```

---

# Part 1: Murder at the Mountain Cabin

A group of old college friends reunited at a remote mountain cabin. By Saturday morning, the host — Marcus Reed — was found dead. The cabin is snowed in. The killer is among the guests.

Your detective agent will interrogate suspects and search locations to identify the killer, the weapon, and the motive.

```python
with open("mystery_data.json") as f:
    cabin_data = json.load(f)

print(f"Mystery: {cabin_data['title']}")
print(f"\n{cabin_data['setting']}")
print(f"\nVictim: {cabin_data['victim']['name']} — {cabin_data['victim']['cause_of_death']}")
print(f"\nSuspects: {', '.join(c['name'] for c in cabin_data['characters'].values())}")
print(f"Locations: {', '.join(loc['name'] for loc in cabin_data['locations'].values())}")
```

### Part 1 Tools

These tools let the detective agent interact with the mystery world. `search_location` returns clues from a location. `interrogate` creates a temporary sub-agent for each suspect — the suspect has its own personality and secrets embedded in a system prompt.

```python
@function_tool
def search_location(location_id: str) -> str:
    """Search a location in the cabin for clues. Valid locations: living_room, kitchen, mudroom, upstairs_hallway, back_porch, marcus_bedroom"""
    location = cabin_data["locations"].get(location_id)
    if not location:
        return f"Unknown location '{location_id}'. Valid: {', '.join(cabin_data['locations'].keys())}"
    log_action("search", f"Searched {location['name']}")
    result = f"**{location['name']}**: {location['description']}\n\nClues found:\n"
    for clue in location["clues"]:
        result += f"- {clue}\n"
    return result


@function_tool
async def interrogate(character_name: str, question: str) -> str:
    """Interrogate a suspect by name. Ask them a specific question. Valid names: diana, larry, tom, sofia"""
    key = character_name.lower().strip()
    character = cabin_data["characters"].get(key)
    if not character:
        return f"Unknown character '{character_name}'. Valid: {', '.join(cabin_data['characters'].keys())}"

    log_action("interrogate", f"Asked {character['name']}: {question[:80]}")

    # Build a system prompt that makes the LLM role-play as this suspect
    system_prompt = f"""You are {character['name']}, {character['role']} in a murder investigation.

Personality: {character['personality']}
Background: {character['background']}
Your secret: {character['secret']}
Your alibi: {character['alibi']}

Things you know or observed:
{chr(10).join('- ' + k for k in character['knows'])}

Behavioral guidelines: {character['guardrails']}

Stay in character. Answer the detective's question based on your knowledge, personality, and secrets. Keep responses to 2-4 sentences. You may lie or deflect according to your behavioral guidelines."""

    suspect_agent = Agent(
        name=character["name"],
        model=AGENTS_MODEL,
        instructions=system_prompt,
    )
    result = await Runner.run(suspect_agent, question)
    response = result.final_output
    log_action("response", f"{character['name']}: {response[:100]}")
    return response

cabin_tools = [search_location, interrogate]
print(f"Tools ready: {[t.name for t in cabin_tools]}")
```

### Part 1 Structured Output

The detective must produce a formal accusation at the end.

```python
class Accusation(BaseModel):
    killer: str
    weapon: str
    motive: str
    evidence: list[str]
```

### TODO 1: Detective Instructions (Part 1)

Write the system prompt for your cabin detective agent. Tell it who it is, what its goal is, and how to approach the investigation (search locations, interrogate suspects, look for contradictions, then make an accusation).

```python
# ✅ SOLUTION
detective_instructions = """You are a seasoned detective investigating the murder of Marcus Reed at a remote mountain cabin. Your goal is to identify the killer, the murder weapon, and the motive.

Investigation strategy:
1. Start by searching the crime scene (living_room) and other key locations for physical evidence.
2. Interrogate all four suspects: Diana, Larry, Tom, and Sofia. Ask pointed questions based on evidence you find.
3. Look for contradictions between suspects' stories and the physical evidence.
4. Follow up on suspicious details — re-interrogate suspects if their stories don't add up.
5. When you have enough evidence, make your accusation with the killer's name, weapon, motive, and supporting evidence.

Be thorough. Search multiple locations and talk to every suspect at least once. Cross-reference what suspects say against physical evidence and other testimony."""
```

### TODO 2: Run Detective Agent (Part 1)

Create the detective `Agent` and run it with `Runner.run()`.

```python
# ✅ SOLUTION
transcript = []  # Reset transcript for Part 1

detective = Agent(
    name="Cabin Detective",
    model=AGENTS_MODEL,
    instructions=detective_instructions,
    tools=cabin_tools,
    output_type=Accusation,
)

result = await Runner.run(
    detective,
    "Investigate the murder of Marcus Reed. Search locations, interrogate all suspects, and identify the killer, weapon, and motive.",
    max_turns=50,
)

accusation = result.final_output
print(f"\n{'='*50}")
print(f"ACCUSATION")
print(f"{'='*50}")
print(f"Killer: {accusation.killer}")
print(f"Weapon: {accusation.weapon}")
print(f"Motive: {accusation.motive}")
print(f"Evidence:")
for e in accusation.evidence:
    print(f"  - {e}")
```

### Save Part 1 Results

```python
part1_results = {
    "killer": accusation.killer,
    "weapon": accusation.weapon,
    "motive": accusation.motive,
    "evidence": accusation.evidence,
    "transcript": transcript,
}

with open("output/part1_results.json", "w") as f:
    json.dump(part1_results, f, indent=2)

print(f"\nSaved to output/part1_results.json")
print(f"Transcript entries: {len(transcript)}")
```

---

# Part 2: Death at St. Mercy Hospital

A structured logic puzzle. Dr. Eleanor Voss was found dead in her office. Security footage shows no one entered or left overnight — the killer was already inside. Use deduction tools to analyze evidence and identify the killer, weapon, and time of death.

```python
with open("mystery_o_matic.json") as f:
    hospital_data = json.load(f)

print(f"Mystery: {hospital_data['title']}")
print(f"\n{hospital_data['setting']}")
print(f"\nVictim: {hospital_data['victim']['name']} — {hospital_data['victim']['cause_of_death']}")
print(f"Suspects: {', '.join(s['name'] for s in hospital_data['suspects'].values())}")
```

### Part 2 Tools

Deduction tools that give the agent access to different categories of evidence.

```python
@function_tool
def get_room_map() -> str:
    """Get the hospital floor map showing all rooms and their locations."""
    log_action("tool", "Retrieved room map")
    result = "Hospital Floor Map:\n"
    for floor, rooms in hospital_data["rooms"].items():
        result += f"\n{floor.replace('_', ' ').title()}: {', '.join(rooms)}"
    return result


@function_tool
def get_witness_statement(suspect: str) -> str:
    """Get a suspect's statement and background. Valid suspects: dr_blake, nurse_chen, dr_santos, orderly_james"""
    key = suspect.lower().strip().replace(" ", "_").replace("dr.", "dr").replace("dr ", "dr_")
    suspect_data = hospital_data["suspects"].get(key)
    if not suspect_data:
        return f"Unknown suspect '{suspect}'. Valid: {', '.join(hospital_data['suspects'].keys())}"
    log_action("interview", f"Got statement from {suspect_data['name']}")
    return f"""**{suspect_data['name']}** ({suspect_data['role']})
Motive: {suspect_data['motive']}
Statement: "{suspect_data['statement']}" """


@function_tool
def get_evidence() -> str:
    """Get all physical evidence and keycard logs from the investigation."""
    log_action("tool", "Retrieved evidence and keycard logs")
    result = "=== KEYCARD ACCESS LOGS ===\n"
    for entry in hospital_data["evidence"]["keycard_logs"]:
        result += f"{entry['time']} | {entry['person']} | {entry['location']} ({entry['action']})\n"
    result += "\n=== PHYSICAL EVIDENCE ===\n"
    for item in hospital_data["evidence"]["physical_evidence"]:
        result += f"- {item}\n"
    result += "\n=== WITNESS OBSERVATIONS ===\n"
    for obs in hospital_data["evidence"]["witness_statements"]:
        result += f"- {obs['witness']}: {obs['statement']}\n"
    return result


@function_tool
def get_weapons() -> str:
    """Get information about potential murder weapons found or available in the hospital."""
    log_action("tool", "Retrieved weapons analysis")
    result = "=== WEAPONS ANALYSIS ===\n"
    for weapon_id, weapon in hospital_data["weapons"].items():
        result += f"\n**{weapon_id.replace('_', ' ').title()}**: {weapon['description']}\n"
        result += f"  Availability: {weapon['availability']}\n"
        result += f"  Notes: {weapon['notes']}\n"
    return result


@function_tool
def get_rules() -> str:
    """Get the logical rules and constraints for solving this mystery."""
    log_action("tool", "Retrieved investigation rules")
    result = "=== INVESTIGATION RULES ===\n"
    for i, rule in enumerate(hospital_data["rules"], 1):
        result += f"{i}. {rule}\n"
    return result

hospital_tools = [get_room_map, get_witness_statement, get_evidence, get_weapons, get_rules]
print(f"Tools ready: {[t.name for t in hospital_tools]}")
```

### Part 2 Structured Output

```python
class PuzzleSolution(BaseModel):
    killer: str
    weapon: str
    time_of_death: str
    reasoning: str
```

### TODO 3: Detective Instructions (Part 2)

Write the system prompt for your hospital detective agent. This is a logic puzzle — emphasize methodical deduction: gather all evidence, check alibis against keycard logs, eliminate suspects, and identify contradictions.

```python
# ✅ SOLUTION
puzzle_instructions = """You are a forensic detective solving the murder of Dr. Eleanor Voss at St. Mercy Hospital. This is a logic puzzle — solve it through methodical deduction.

Investigation approach:
1. First, get the rules and room map to understand the constraints.
2. Gather all evidence: keycard logs, physical evidence, and witness observations.
3. Get each suspect's statement and compare it against the hard evidence (keycard logs don't lie).
4. Check which suspects had access to Floor 4 during the time of death window (9:00-10:00 PM).
5. Determine which weapon matches the cause of death.
6. Look for contradictions between statements and evidence — the killer's alibi will not match the keycard logs.
7. Provide your solution with clear logical reasoning."""
```

### TODO 4: Run Detective Agent (Part 2)

Create the hospital detective `Agent` and run it with `Runner.run()`.

```python
# ✅ SOLUTION
transcript = []  # Reset transcript for Part 2

hospital_detective = Agent(
    name="Hospital Detective",
    model=AGENTS_MODEL,
    instructions=puzzle_instructions,
    tools=hospital_tools,
    output_type=PuzzleSolution,
)

result = await Runner.run(
    hospital_detective,
    "Solve the murder of Dr. Eleanor Voss. Use all available tools to gather evidence, analyze suspects, and determine the killer, weapon, and time of death.",
    max_turns=15,
)

solution = result.final_output
print(f"\n{'='*50}")
print(f"SOLUTION")
print(f"{'='*50}")
print(f"Killer: {solution.killer}")
print(f"Weapon: {solution.weapon}")
print(f"Time of death: {solution.time_of_death}")
print(f"Reasoning: {solution.reasoning}")
```

### Save Part 2 Results

```python
part2_results = {
    "killer": solution.killer,
    "weapon": solution.weapon,
    "time_of_death": solution.time_of_death,
    "reasoning": solution.reasoning,
    "transcript": transcript,
}

with open("output/part2_results.json", "w") as f:
    json.dump(part2_results, f, indent=2)

print(f"\nSaved to output/part2_results.json")
print(f"Transcript entries: {len(transcript)}")
```

---

# Validation

Run the tests to check your results:

```python
!python -m pytest .github/tests/ -v
```

---

# Part 3: Interactive Interrogation (Optional, Not Graded)

Want to interrogate the cabin suspects yourself? This chat loop lets you play detective interactively.

```python
interactive_detective = Agent(
    name="Interactive Detective",
    model=AGENTS_MODEL,
    instructions="You are assisting a human detective investigating the murder of Marcus Reed at a mountain cabin. The human will tell you who to interrogate or where to search. Use your tools to carry out their requests and report back what you find.",
    tools=cabin_tools,
)

print("=== Interactive Detective Mode ===")
print("Tell the detective who to interrogate or where to search.")
print("Examples: 'Ask Diana about her alibi', 'Search the kitchen', 'Interrogate Larry about the treasure map'")
print("Type 'quit' to exit.\n")

while True:
    user_input = input("Detective> ").strip()
    if user_input.lower() in ("quit", "exit", "q"):
        print("Case closed.")
        break
    if not user_input:
        continue
    result = await Runner.run(interactive_detective, user_input)
    print(f"\n{result.final_output}\n")
```
