# Lecture Review Prompts

Reusable prompts for subagent-based lecture reviews.

## Usage

Launch 3 parallel subagents with these prompts to review a lecture:

```
1. Content completeness: lecture_review_content.md
2. Media & visuals: lecture_review_media.md
3. Beginner accessibility: lecture_review_accessibility.md
```

## Prompt Template Structure

Each prompt includes:
- **AGENTS.md citations**: Specific requirements from the authoring guide
- **Specific checks**: Concrete things to look for
- **Output format**: Structured findings for easy synthesis

## Customization

Prompts reference `AGENTS.md` and course structure. Update prompts if:
- AGENTS.md requirements change
- New lecture format/structure adopted
- Different audience or course level
