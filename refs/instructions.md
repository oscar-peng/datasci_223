## Meta-Instruction

This file is supplemental guidance and examples for authoring lectures in this repo.

- **Source of truth:** If anything in this file conflicts with `AGENTS.md`, follow `AGENTS.md`.
- **Audience:** Health data science masters students who are beginners in programming (Python, git, shell commands).
- **Continuous evaluation:** Before generating each section, ensure content matches student level and balances concept / reference / example.
- **Time structure:** Design for 90-minute lectures (plus demos), typically as long-form Markdown.
- **No time estimates in text:** Do not include time cues/estimates in lecture/demo/assignment content.
- **Demo integration:** Include 3 hands-on demo breaks (at ~⅓, ~⅔, and end points).

## Core Principles

### 1. Format & Structure

- **Markdown format:** Create lectures in long-form Markdown. Only if requested, use Marp markdown slide format (separated by `---`).
- **Speaking notes (optional):** Put talking points in `NOTES.md` (same section headings), after the lecture content stabilizes.
- **Progressive learning:** Build knowledge incrementally.
- **Visual organization:** Use consistent heading levels, bullets, and whitespace.

Example section (complex topics may span multiple sections; include at least one reference card and one code snippet per major concept):

"""

# Classifiers

## `RandomForestClassifier`

Random forests are a robust baseline for many tabular health-data tasks. They combine many decision trees trained on random subsets of data and vote on the final prediction.

General flow:

1. **Bagging**: create *k* random samples from the dataset
2. **Grow trees**: build decision trees using splits that separate classes
3. **Classify**: vote across trees for a final prediction

![Random Forest Visualization](media/random_forest.png)

### Reference Card: `RandomForestClassifier`

- **Function:** `sklearn.ensemble.RandomForestClassifier()`
- **Purpose:** Ensemble of decision trees for classification
- **Key Parameters:**
    - `n_estimators`: (Optional, default=100) number of trees
    - `max_depth`: (Optional, default=None) maximum depth of each tree
    - `random_state`: (Optional, default=None) controls randomness for reproducibility

### Code Snippet: `RandomForestClassifier`

```python
from sklearn.ensemble import RandomForestClassifier

X = [[1, 2], [2, 3], [3, 4], [4, 5]]
y = [0, 1, 0, 1]

model = RandomForestClassifier(n_estimators=10).fit(X, y)
print(model.predict([[2, 2]]))
```

![xkcd: Scenario 4](media/xkcd_scenario_4.png)

"""

### 2. Content Balance

- **Conceptual foundations:** Explain how things work in accessible terms.
- **Reference material:** Include function definitions, syntax rules, and common parameters.
- **Practical examples:** Provide brief example code samples inline with little-to-no context.
- **Hands-on demos:** Apply methods to realistic health data with more complexity than lecture snippets.

### 3. Teaching Approach

- **Beginner-friendly:** Avoid jargon; define terms when introduced.
- **Visual learning:** Prefer diagrams, screenshots, and concrete outputs.
- **Misconceptions:** Address common beginner mistakes in `NOTES.md` (or in tightly-scoped callouts) when it improves clarity.

### 4. Tone, Humor, and Visual Cues

- **Not a script:** The lecture text is not speaking notes; it should read like reference material + worked examples.
- **Humor:** Sprinkle relevant humor *between* sections/sub-sections. Avoid making the core explanation itself jokey in a way that obscures meaning.
- **Emojis:** Use sparingly as visual anchors; avoid emoji-only meaning.
- **Code annotations:** Short comments in code snippets are fine when they clarify intent.

### 5. Demo Break Structure

- **Hands-on learning:** Design 3 practical demo sessions at roughly ⅓, ⅔, and the conclusion of the lecture.
- **Progressive difficulty:** Start simple, build complexity across demos.
- **Clear instructions:** Provide step-by-step guidance with expected outcomes.
- **Success validation:** Include ways to confirm completion (visual/tabular output after each step).
- **Markdown/Jupytext format:** Most demos will be Jupyter notebooks; write them in Markdown and convert with `jupytext`.
- **Demo headings stay clean:** In the lecture, `# LIVE DEMO!` headings only mark the break—put walkthrough steps inside `lectures/XX/demo/`.
