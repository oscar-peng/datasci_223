# Assignment 8 Hints

## TODO 1 & 3: System Prompts

Your system prompt tells the agent *how* to investigate. A good prompt includes:

- **Role**: Who is the agent? (e.g., "You are a detective investigating...")
- **Goal**: What should it figure out? (killer, weapon, motive)
- **Strategy**: How should it approach the investigation?
  - Part 1: Search locations, interrogate all suspects, look for contradictions, follow up
  - Part 2: Gather all evidence first, compare statements to keycard logs, eliminate suspects
- **When to conclude**: Make an accusation only after gathering sufficient evidence

Example structure (don't copy verbatim — write your own):
```
You are a [role] investigating [case].
Your goal is to determine [what].
Strategy: [steps].
[Any other guidance].
```

## TODO 2 & 4: Creating and Running Agents

The pattern is the same for both parts:

```python
agent = Agent(
    name="Some Name",
    model=AGENTS_MODEL,
    instructions=your_instructions_variable,
    tools=your_tools_list,
    output_type=YourPydanticModel,
)

result = await Runner.run(agent, "Your task message here", max_turns=N)
output = result.final_output
```

Key parameters:
- `output_type` forces the agent to return structured data matching your Pydantic model instead of free text. The agent keeps calling tools until it has enough info, then returns a parsed object. Access fields directly: `result.final_output.killer`, `result.final_output.weapon`, etc.
- `max_turns` limits how many tool calls the agent can make (50 for Part 1, 15 for Part 2)
- The task message should tell the agent what to investigate

## Part 1 Tips

- The agent needs to search **multiple locations** — the crime scene alone isn't enough
- Interrogate **all four suspects** — key information is spread across different people
- The killer's story will have **contradictions** with physical evidence
- If the agent doesn't find the right answer, try making your system prompt more specific about investigation strategy

## Part 2 Tips

- This is a **logic puzzle** — there's one definitive answer
- The keycard logs are the **most important evidence** — they can't be faked
- Compare each suspect's **statement** against their **keycard activity**
- The cause of death tells you **which weapon** was used
- One suspect's alibi is contradicted by the keycard logs — that's your killer

## Common Issues

- **Module not found**: Run `pip install -r requirements.txt`
- **API key not found**: Make sure `.env` has `OPENROUTER_API_KEY=...` (no quotes around the value)
- **Agent runs out of turns**: Increase `max_turns` or make your prompt more focused
- **Wrong answer**: Improve your system prompt — the agent needs clear instructions to investigate thoroughly
- **NameError for `accusation`/`solution`**: Make sure your TODO 2/4 cell assigns `result.final_output` to the expected variable name
