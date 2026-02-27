# Assignment 8 Hints

## The Iteration Loop

The core skill in this assignment is **prompt engineering through iteration**. You don't need to know the mystery's answer — you need to write instructions that make the agent investigate thoroughly enough to find it. The loop is:

1. Write instructions → run the agent → check the output
2. If the output is wrong or incomplete, look at the intermediate results (the checkpoints help)
3. Adjust your instructions → re-run → check again

Each stage's structured output (`evidence`, `findings`, `hospital_evidence`) is inspectable — print it to see exactly what the next agent will receive.

## TODO 1: Stage 1 — Crime Scene Investigation

Create an agent that searches all six locations. Your instructions should:

- Give the agent a role ("You are a crime scene investigator...")
- **List all six location IDs explicitly**: `living_room`, `kitchen`, `mudroom`, `upstairs_hallway`, `back_porch`, `marcus_bedroom`
- Tell it to summarize what it finds

```python
search_agent = Agent(
    name="Crime Scene Investigator",
    model=AGENTS_MODEL,
    model_settings=SETTINGS,
    instructions="...",  # Tell it to search ALL six locations and summarize
    tools=[search_location],
    output_type=EvidenceSummary,
)
search_result = await Runner.run(search_agent, "Search all six locations for evidence.", max_turns=15)
evidence = search_result.final_output
```

**How to check:** You should see 6 `display_search` cards (one per location) and `evidence.clues` should have 10+ items. If fewer, your instructions didn't list all locations.

## TODO 2: Stage 2 — Suspect Interrogation

The key idea: **embed Stage 1's output in Stage 2's instructions**. This is how results flow between agents in a prompt chain.

```python
evidence_brief = f"""Evidence found so far:
Clues: {chr(10).join('- ' + c for c in evidence.clues)}
Suspicious items: {', '.join(evidence.suspicious_items)}
Key questions: {chr(10).join('- ' + q for q in evidence.key_questions)}"""

interview_agent = Agent(
    name="Lead Interrogator",
    model=AGENTS_MODEL,
    model_settings=SETTINGS,
    instructions=f"""You are a detective interrogating murder suspects.

{evidence_brief}

Interview each suspect (diana, larry, tom, sofia). For each suspect:
1. Ask about their alibi and whereabouts during the murder
2. Follow up on anything suspicious — if their answer seems evasive,
   contradicts the physical evidence, or doesn't explain a suspicious item,
   press them with a second question

After all interviews, summarize your findings.""",
    tools=[interrogate],
    output_type=InterviewFindings,
)
```

**How to check:** You should see 4+ chat bubble pairs (initial questions + follow-ups). `findings.contradictions` should NOT be empty — if it is, your instructions aren't directing the agent to ask about the suspicious items. Try telling it specifically: "Follow up if their answer contradicts physical evidence."

## TODO 3: Stage 3 — Final Deduction

Combine Stage 1 + Stage 2 outputs into a `case_file` string and pass it to a reasoning-only agent (no tools):

```python
case_file = f"""{evidence_brief}

Interview findings:
Alibis: {chr(10).join('- ' + a for a in findings.alibis)}
Contradictions: {chr(10).join('- ' + c for c in findings.contradictions)}
Key observations: {chr(10).join('- ' + o for o in findings.key_observations)}"""

deduction_agent = Agent(
    name="Lead Detective",
    model=AGENTS_MODEL,
    model_settings=SETTINGS,
    instructions="...",  # Compare alibis against physical evidence — whose story is contradicted?
    output_type=Accusation,  # No tools — just reasoning
)
deduction_result = await Runner.run(deduction_agent, f"Make your accusation:\n\n{case_file}", max_turns=3)
accusation = deduction_result.final_output
```

**How to check:** If the accusation is wrong, `print(case_file)` to see what the deduction agent received. If the contradictions are weak or missing, go back and improve TODO 2's instructions. The deduction agent can only work with what it's given.

## TODO 4: Hospital Mystery (Two Stages)

Same chaining pattern as Part 1 — split into evidence gathering and deduction.

**Stage 1 (Evidence Collector):** An agent with all five tools that gathers everything. Tell it to get rules, room map, evidence, weapons, and ALL four suspect statements (`dr_blake`, `nurse_chen`, `dr_santos`, `orderly_james`). Returns `HospitalEvidence`.

**Stage 2 (Solver):** A reasoning-only agent (no tools) that receives the evidence dossier and deduces the answer. Your solver instructions should include these deduction principles:
- Keycard logs are hardware records — they **cannot be faked**
- Compare each suspect's **statement** against their **keycard activity** — the liar is the killer
- Match the **weapon** to the **cause of death**
- For **time of death**: the victim was still alive at the time of her last activity (email). The murder happened **after** that. Look for a **witness observation** that places someone near the victim's room at a specific time — that's when it happened

## Debugging: If the Agent Gets It Wrong

**Part 1:**
- Print `evidence.clues` — are there clues from all 6 locations?
- Print `findings.contradictions` — did the interrogator find any? If empty, your Stage 2 instructions need to be more specific about what to ask
- Print `case_file` — does it contain enough information for a human to solve it?
- Try the Interactive Interrogation section (Part 5) to manually question suspects and understand the mystery

**Part 2:**
- Print `hospital_evidence.witness_observations` — does it include a specific time and location?
- Print `hospital_evidence.keycard_logs` — are all entries preserved with times?
- Print `evidence_dossier` — can YOU solve it from reading it? If not, the agent can't either
- The most common error is wrong time of death — that comes from witness observations, not the victim's last email

## Common Issues

- **Module not found**: Run `pip install -r requirements.txt`
- **API key not found**: Make sure `.env` has `OPENROUTER_API_KEY=...` (no quotes around the value)
- **Agent runs out of turns**: Increase `max_turns` or make your prompt more focused
- **Wrong answer**: Improve your system prompt — check intermediate outputs to see where the chain broke down
- **NameError for `evidence`/`findings`/`accusation`/`solution`**: Make sure each TODO cell assigns the expected variable name
