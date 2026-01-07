## Meta-Instruction

- **Audience Assessment:** Create content for health data science masters students who are beginners in programming (Python, git, shell commands)
- **Continuous Evaluation:** Before generating each section, ensure content matches student level, balances concept / reference / example, and includes speaking notes
- **Time Structure:** Design for 90-minute lectures, with maximum 60 slides if using Marp. Otherwise, long-form Markdown.
- **Demo Integration:** Include 3 hands-on demo breaks (at ⅓, ⅔, and end points)

## Core Principles

### 1. Format & Structure

- **Markdown Format:** Create lectures in long-form Markdown with valid markdownlint EXCEPT for Notion-specific quirks as noted in AGENTS.md . Only if requested, use marp markdown slide format (separated by `---`).
- **Speaking Notes:** Talking points for each lecture should be placed in NOTES.md in the same directory and with matching section headings. The talking points should be supplemental content/context to each heading to sub-sub-heading (# -> ###), written to be helpful to students reading the lecture notes on their own. It should _NOT_ be instructions to the lecturer; e.g., "speak about X" or "mention relationship to Y". These notes should be concise and focused on the key points of the lecture and be created only after the lecture is complete.
- **Progressive Learning:** Structure content to build knowledge incrementally
- **Visual Organization:** Use consistent heading levels, bullet points, and white space

Example section (complex topics may span multiple sections, only need the reference card/example at least once):
#### Random forest

"""
# Classifiers

## `RandomForestClassifier`

Random forests are powerful and robust for tabular health data. They combine many decision trees, each trained on a random subset of the data, and vote on the final prediction. Beginners sometimes think more trees always means better results, but too many can slow things down.

Each of the steps can be tweaked, but the general flow goes:

1. **Bagging** - create _k_ random samples from the data set
2. **Grow trees** - individual decision trees are constructed by choosing the best features and cutpoints to separate the classes
3. **Classify** - instances are run through all trees and assigned a class by majority vote


![Random Forest Visualization](media/random_forest.png)

### Reference Card: `RandomForestClassifier`

- **Function:** `sklearn.ensemble.RandomForestClassifier()`
- **Purpose:** Ensemble of decision trees for classification
- **Key Parameters:** 
  - `n_estimators`: (Optional, default=100) The number of trees in the forest. More trees generally improve performance but increase computation time
  - `max_depth`: (Optional, default=None) The maximum depth of each tree. If `None`, nodes are expanded until all leaves are pure or contain less than `min_samples_split` samples. Deeper trees can capture more complex patterns but risk overfitting
  - `random_state`: (Optional, default=None) Controls the randomness of the bootstrapping of the samples used when building trees and the sampling of the features to consider when looking for the best split at each node. Setting a specific number ensures reproducibility
  - `parameter`: (Optional/Required, default=XXX) Brief summary of what `parameter` is and does in a sentence or two. No commentary.
  
### Code Snippet: `RandomForestClassifier`

```python
from sklearn.ensemble import RandomForestClassifier
X = [[1, 2], [2, 3], [3, 4], [4, 5]]
y = [0, 1, 0, 1]
model = RandomForestClassifier(n_estimators=10).fit(X, y)
print(model.predict([[2, 2]]))
```

![xkcd 2289: Scenario 4](https://imgs.xkcd.com/comics/scenario_4.png)
--->
"""

### 2. Content Balance

- **Conceptual Foundations:** Explain how things work in accessible terms
- **Reference Material:** Include function definitions, syntax rules, and common parameters
- **Practical Examples:** Provide brief example code samples inline with little-to-no context
- **Hands-on Demos:** Should wrap up major section, ideally applying methods with health data applications with more complexity and depth than the code snippets from lecture.

### 3. Teaching Approach

- **Beginner-Friendly:** Avoid jargon, explain terms when introduced
- **Visual Learning:** Use diagrams, analogies, screenshots, and concrete examples/outputs
- **Engagement:** Include comprehension checkpoints and practice opportunities
- **Misconceptions:** Address common beginner mistakes in speaking notes

### 4. Tone & Style

- **Professional but Engaging:** Maintain educational focus while being approachable
- **Strategic Humor:** Include occasional nerdy puns (xkcd-style) and cheesy pop culture references (80s/90s movies)
- **Visual Cues:** Use emoji and formatting to highlight key points and create visual interest
- **Clear Annotations:** Comment key lines within code examples

### 5. Demo Break Structure

- **Hands-On Learning:** Design 3 practical demo sessions (10-15 minutes each) at roughly 1/3, 2/3, and conclusion of lecture
- **Progressive Difficulty:** Start simple, build complexity across demos. Difficulty should always stay accessibly understandable by students learning the topic.
- **Clear Instructions:** Provide step-by-step guidance with expected outcomes
- **Success Validation:** Include ways to confirm students completed tasks correctly, e.g., visual or tabular output after every code section.
- **Markdown/Jupytext Format:** Most demos with by Jupyter notebooks, but we will write them using markdown and convert with `jupytext`.
