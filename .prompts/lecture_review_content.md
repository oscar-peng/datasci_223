# Lecture Content Review Prompt

Review the lecture for **content completeness** against AGENTS.md requirements:

## Check Against AGENTS.md

1. **Content style** (from AGENTS.md):
   > "each section/subsection should include (1) brief prose intro/explanation, (2) optional visual or `#FIXME` placeholder, (3) reference (function signature, common parameters), and (4) minimal code example."

   - Does each subsection have all four elements?
   - Are there sections missing reference documentation?
   - Are code examples minimal and focused?

2. **Speaking notes** (from AGENTS.md):
   > "include HTML comment speaking notes under each `###`"

   - Does each `###` heading have a `<!--- --->` speaking note?
   - Are speaking notes supplemental guidance (not instructional content)?

3. **Demo format** (from AGENTS.md):
   > "Demos should have more complexity with real/realistic data"

   - Are demo sections clearly marked with timing (e.g., "Demo (~30 min)")?
   - Do demos use realistic health data examples?

## Specific Content Gaps to Check

- Is there coverage of Windows/WSL2 setup alongside macOS/Linux?
- Are all jargon terms defined on first use (DRY, KISS, pdb, CLI flags, etc.)?
- Does the progression go from simple → complex (e.g., print debugging before pdb)?
- Are there any topics mentioned in images/media that aren't covered in text?

## Output Format

List findings as:
- **Missing**: Content required by AGENTS.md but absent
- **Incomplete**: Content present but missing elements (visual, reference, example)
- **Suggestions**: Optional improvements for clarity or completeness
