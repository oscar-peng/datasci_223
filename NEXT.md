# Next steps: Lecture 01 polish

## Blocking fixes

- Add the actual GitHub Classroom invite URL and a Classroom acceptance screenshot.
- Replace VS Code debugger placeholders with real captures (breakpoint gutter/call stack and notebook debug controls).

## Content improvements (instructional)

- Spell out the assignment tasks: name the starter repo/notebook/script, describe required edits (e.g., add bounds checks/logging, fix BMI bugs, document a VS Code debug walkthrough), required outputs, and how to run checks locally (`pytest .github/tests -q`). Include a short submission checklist.
- Link demos to the assignment explicitly: note whether the assignment reuses the BMI script/notebook from the demos or point to the specific assignment files.
- Specify the starter notebook filename in the “Accept/open starter repo” demo and include a quick verification step (e.g., Run All succeeds, key cells output).
- Enrich the VS Code demo steps: list the intentional bugs (formula/typo/indexing) students will see, and show expected output before/after fixes.
- Trim meta padding: keep summaries concise; favor concrete, domain-relevant steps (sample log output, example assertion) over generic Signature/Example boilerplate where it doesn’t add clarity.

## Optional additions

- If needed, fetch more relevant XKCD panels via `./scripts/fetch_xkcd_2x.py <id:Slug[:filename]>` and swap into sections for variety.
