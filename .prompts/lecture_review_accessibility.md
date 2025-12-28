# Lecture Beginner Accessibility Review Prompt

Review the lecture for **beginner accessibility** against course requirements:

## Check Against AGENTS.md

1. **Audience** (from AGENTS.md):
   > "audience is beginner health data science students"

   - Is jargon explained on first use?
   - Are there unexplained acronyms or technical terms?
   - Do code examples start simple and build complexity?

2. **Course structure** (from AGENTS.md):
   > "90-minute lectures with additional demo time at 1/3, 2/3, and end"

   - Are there clear demo breaks at ~30, ~60, and ~90 minutes?
   - Do demos give students hands-on practice opportunities?

## Specific Accessibility Checks

1. **Unexplained jargon**: List any technical terms used without definition
   - Examples: DRY, KISS, pdb, CLI flags, pure functions, virtual environments
   - Check if `-m` flag, `venv`, `pip`, etc. are explained

2. **Code complexity jumps**: Identify places where examples jump difficulty
   - Does the lecture start with simple examples?
   - Are there intermediate steps between simple and complex?
   - Would a beginner understand each code example without prior context?

3. **Platform coverage**: Check OS/environment coverage
   - Are Windows, macOS, and Linux all covered?
   - Are Windows-specific differences noted (e.g., `.venv\Scripts\activate`)?
   - Is WSL2 mentioned as an option for Windows users?

4. **Comprehension checkpoints**: Look for learning verification
   - Are there "Try it yourself" boxes?
   - Are there knowledge checks or reflection questions?
   - Do demos include verification steps (e.g., "run this to confirm it works")?

## Output Format

- **Jargon**: List unexplained terms with first occurrence location
- **Code jumps**: List examples that may be too complex for beginners
- **Platform gaps**: Note any OS-specific missing guidance
- **Suggestions**: Ways to add comprehension checkpoints or simplify complex sections
