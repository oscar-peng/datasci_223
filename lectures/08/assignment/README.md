# Assignment 8: Murder Mystery Agents

Build detective agents that solve two murder mysteries using the OpenAI Agents SDK.

## Mysteries

- **Part 1: Murder at the Mountain Cabin** — A narrative mystery with LLM-powered suspects. Your agent interrogates characters and searches locations to solve the case.
- **Part 2: Death at St. Mercy Hospital** — A structured logic puzzle. Your agent uses deduction tools to analyze evidence and identify the killer.

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

## Workflow

1. Open `assignment.ipynb` and work through the notebook
2. Complete 4 TODO cells:
   - **TODO 1**: Write the detective system prompt for Part 1
   - **TODO 2**: Create and run the detective Agent for Part 1
   - **TODO 3**: Write the detective system prompt for Part 2
   - **TODO 4**: Create and run the detective Agent for Part 2
3. Run all cells to generate output files

## Output Files

Tests check these artifacts (generated when you run the notebook):
- `output/part1_results.json` — Part 1 accusation (killer, weapon, motive, evidence, transcript)
- `output/part2_results.json` — Part 2 solution (killer, weapon, time of death, reasoning, transcript)

## Testing

```bash
python -m pytest .github/tests/ -v
```

Tests verify your agent identified the correct answers and actually used its tools (not just hardcoded answers).

## Tips

- The agent needs good instructions to investigate thoroughly — vague prompts lead to incomplete investigations
- Part 1 may need multiple runs if the agent doesn't find all the clues (LLMs are non-deterministic)
- Part 2 is deterministic — the logic puzzle has one correct answer
- See `hints.md` if you get stuck
