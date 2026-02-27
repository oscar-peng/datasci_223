# Assignment 8: Murder Mystery Agents

Build detective agent workflows that solve two murder mysteries using the OpenAI Agents SDK and **prompt chaining** — each agent handles one stage of the investigation and passes structured results to the next.

## Mysteries

- **Part 1: Murder at the Mountain Cabin** — A narrative mystery with LLM-powered suspects. Your agents search locations, interrogate characters, and deduce the killer across three stages.
- **Part 2: Death at St. Mercy Hospital** — A structured logic puzzle. Your agents gather evidence and use deduction to identify the killer, weapon, and time of death.

## Setup

```bash
pip install -r requirements.txt
```

### API Key

You need an OpenRouter API key (or OpenAI key) for this assignment. Get the shared key from the class forum, or create a free account at [openrouter.ai](https://openrouter.ai).

Create a `.env` file:
```
OPENROUTER_API_KEY=your_key_here
```

**WARNING**: Do not commit your `.env` file — your key will be invalidated for all students.

## What You'll Do

Open `assignment.ipynb` and write the `instructions` string for each Agent. The tools, structured outputs, and display functions are all provided — your job is to write prompts that make each agent investigate effectively.

**4 TODOs across 2 mysteries:**

| TODO | Agent | What it does | Your instructions should... |
| ---- | ----- | ------------ | -------------------------- |
| 1 | Crime Scene Investigator | Searches cabin locations | List all 6 locations, tell it to summarize |
| 2 | Lead Interrogator | Questions suspects | Include evidence from Stage 1, direct follow-up questions |
| 3 | Lead Detective | Makes the accusation | Tell it to compare alibis against physical evidence |
| 4 | Evidence Collector + Solver | Gathers hospital evidence, then deduces | List all tools/suspects; reason from keycard logs |

The notebook includes an **interactive detective** you can use to manually explore the mystery — search locations and interrogate suspects to understand the evidence before writing your agent instructions.

## Output Files

Tests check these artifacts (generated when you run the notebook):
- `output/part1_results.json` — Part 1 accusation (killer, weapon, motive, evidence, transcript)
- `output/part2_results.json` — Part 2 solution (killer, weapon, time of death, reasoning, transcript)

## Testing

```bash
python -m pytest .github/tests/ -v
```

Tests verify your agents identified the correct answers and actually used their tools.

## Tips

- **Iterate on your instructions** — run the agent, check the output, improve the prompt, re-run
- Use the checkpoint cells and `display_notes` panels to inspect what each stage produced
- If an agent gets the wrong answer, the issue is usually in an earlier stage's instructions
- Part 1 is non-deterministic — multiple runs may give slightly different results
- Part 2 is a logic puzzle with one correct answer — the keycard logs are the key
- See `hints.md` if you get stuck
