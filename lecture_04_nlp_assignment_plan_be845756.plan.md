---
name: Lecture 04 NLP Assignment Plan
overview: "Create a pass/fail assignment for Lecture 04 NLP framed as a medical investigation mystery. Students analyze patient case reports using NLP techniques (preprocessing, POS/NER/regex extraction, document comparison, pipeline building) to solve 'The Symptom Cluster Investigation'. Assignment follows AGENTS.md guidelines: test behaviors/artifacts, not implementation details."
todos:
  - id: assignment-structure
    content: Create assignment directory structure (README.md, source files, .github/tests/, data/)
    status: pending
  - id: sample-data
    content: Create 3-4 engaging patient case reports with narrative style and embedded mystery clues in data/ directory
    status: pending
  - id: part1-preprocessing
    content: Create preprocess_reports.py with preprocess_text() function and starter code
    status: pending
  - id: part2-extraction
    content: Create extract_clues.py with extract_clues() function and starter code
    status: pending
  - id: part3-comparison
    content: Create compare_cases.py with compare_cases() function and starter code
    status: pending
  - id: part4-pipeline
    content: Create solve_mystery.py with analyze_case() function and starter code
    status: pending
  - id: test-suite
    content: Create .github/tests/test_assignment.py with behavior-based tests for all parts
    status: pending
  - id: readme-instructions
    content: Write README.md with clear instructions, requirements, and success criteria
    status: pending
  - id: requirements-files
    content: Create requirements.txt and .github/tests/requirements.txt
    status: pending
  - id: github-actions
    content: Create .github/workflows/classroom.yml for autograding
    status: pending
  - id: validation
    content: "Test assignment from scratch: copy to temp dir, run student workflow, verify tests pass"
    status: pending
isProject: false
---

# Lecture 04 NLP Assignment Plan

## Assignment Theme: The Symptom Cluster Investigation

**Narrative:** Students are investigating a cluster of unusual cases. Multiple patients have presented with similar symptoms, and the team needs to analyze free-text clinical reports to find the common thread. Each part of the assignment uses NLP to uncover clues that lead to identifying the connection.

**Inspiration:** Similar to the SQL Murder Mystery (lecture 03), but focused on medical investigation using NLP techniques. The mystery unfolds as students apply preprocessing, extraction, comparison, and pipeline building.

## Assignment Structure

Following the pattern from `lectures/01/assignment/`:

```
lectures/04/assignment/
├── README.md                    # Instructions with investigation narrative
├── preprocess_reports.py        # Part 1: Clean the evidence (preprocessing)
├── extract_clues.py             # Part 2: Find the patterns (POS/NER/regex)
├── compare_cases.py             # Part 3: Connect the dots (document comparison)
├── solve_mystery.py             # Part 4: Build the case (pipeline)
├── data/                        # Patient case reports
│   ├── case_001.txt            # Patient report 1
│   ├── case_002.txt            # Patient report 2
│   ├── case_003.txt            # Patient report 3
│   └── case_004.txt            # Patient report 4 (optional)
├── requirements.txt             # nltk, spacy, scikit-learn, pyyaml
├── hints.md                     # Optional hints (investigation tips)
└── .github/
    ├── tests/
    │   ├── test_assignment.py   # pytest test suite
    │   └── requirements.txt      # pytest, jupyter, nbconvert
    └── workflows/
        └── classroom.yml        # GitHub Actions autograding
```

## Data: Patient Case Reports

Provide 3-4 engaging patient case reports (synthetic, not real patient data) written in a narrative style that mirrors clinical notes but with a story element. Each case should:

- Have a patient name and visit date
- Describe symptoms in natural language (not just lists)
- Include vitals (BP, HR, temp) embedded in sentences
- Mention medications with dosages
- Use negation patterns ("denies chest pain", "no history of")
- Include abbreviations and variations (BP, blood pressure, b.p.)
- Have subtle connections to other cases (common symptom, medication, timeline, location)

**Example style:**

```
Patient: Dr. Sarah Chen
Date: March 15, 2024

Dr. Chen presented to the emergency department complaining of 
severe headaches that began three days ago. She denies any 
history of migraines. Blood pressure was elevated at 150/95. 
Patient reports taking Ibuprofen 400mg twice daily with minimal 
relief. Temperature 98.6F, heart rate 88 bpm. Patient works as 
a researcher at the university lab. Recent travel to conference 
in Boston last week. No known drug allergies.
```

Each report ~200-400 words. The "mystery" could be:

- All patients took the same medication (adverse event)
- All visited the same location/event (outbreak)
- All have a symptom pattern that suggests a diagnosis
- Timeline reveals when the issue started

**Example Mystery Scenarios:**

1. **The Medication Mystery:** All patients mention taking "NewVital 500mg daily" (or similar). The distinctive terms in TF-IDF would highlight this medication. Extraction would show it in all cases.
2. **The Event Mystery:** All patients attended "the annual research symposium" or visited "the new wing of the hospital" in the past week. NER might catch location names, or distinctive terms would reveal the connection.
3. **The Symptom Pattern:** All patients describe "severe fatigue" and "muscle weakness" (variations: "tired", "exhausted", "weakness", "aching muscles"). Preprocessing and TF-IDF would group these variations.
4. **The Timeline Clue:** Dates show all patients visited within 48 hours of each other, suggesting a point-source exposure. The extraction function would reveal the tight timeline.

Store in `data/` directory as `case_001.txt`, `case_002.txt`, etc.

## Part 1: Clean the Evidence

**File:** `preprocess_reports.py`

**Narrative:** The case reports are messy—full of variations, abbreviations, and inconsistencies. Before we can analyze them, we need to normalize the text. Your preprocessing function will clean the evidence so patterns become visible.

**Task:** Implement a preprocessing function that tokenizes, normalizes, and lemmatizes text.

**Requirements:**

- Function signature: `preprocess_text(text: str, lowercase: bool = True, remove_stopwords: bool = True) -> list[str]`
- Use `nltk.word_tokenize()` for tokenization
- Apply lowercase if `lowercase=True`
- Remove stopwords if `remove_stopwords=True` (but preserve "no" and "not" for negation)
- Apply lemmatization using `WordNetLemmatizer`
- Return list of processed tokens

**Investigation note:** Pay attention to how preprocessing affects medical terms. "BP" and "blood pressure" should both appear in your processed tokens (if you handle abbreviations), but lemmatization helps group "headaches" and "headache" together.

**Test behavior:**

- Function executes without errors
- Returns list of strings
- Known input/output pairs: test with sample text, verify tokens are lemmatized
- Stopwords removed (except "no"/"not")
- Lowercase applied when flag is True

**Artifact:** Function in `preprocess_reports.py` that can be imported and tested

## Part 2: Find the Patterns

**File:** `extract_clues.py`

**Narrative:** Now that the text is cleaned, we need to extract key information: who, when, what medications, and what vitals. These are the clues that might connect the cases. Use POS tagging, named entity recognition, and regex to pull out structured data from the free text.

**Task:** Extract structured information from clinical-style text using POS, NER, and regex.

**Requirements:**

- Function: `extract_clues(text: str) -> dict`
- Use spaCy for POS and NER (load model once, reuse)
- Extract:
  - Patient names (PERSON entities from NER)
  - Dates (DATE entities from NER + regex for numeric dates like "03/15/2024")
  - Blood pressure readings (regex: `\d{2,3}/\d{2,3}`)
  - Medication dosages (regex: medication name + `\d+\s?(mg|ml|mcg)`)
- Return dict with keys: `names`, `dates`, `blood_pressure`, `medications`

**Investigation note:** Look for patterns in the extracted data. Do all patients have similar BP? Are they all on the same medication? The dates might reveal a timeline.

**Test behavior:**

- Function executes without errors
- Returns dict with expected keys
- Known input/output: test with sample text containing known entities
- NER finds PERSON entities
- Regex extracts BP patterns and medication dosages
- Dates extracted (both NER and regex)

**Artifact:** Function in `extract_clues.py` that produces structured output

## Part 3: Connect the Dots

**File:** `compare_cases.py`

**Narrative:** Which cases are most similar? By comparing the documents using TF-IDF, we can find distinctive terms in each case and measure how similar they are. The most similar cases might share the same underlying cause.

**Task:** Compare multiple documents using TF-IDF and cosine similarity.

**Requirements:**

- Function: `compare_cases(file_paths: list[str]) -> dict`
- Load all text files from paths
- Vectorize using `TfidfVectorizer(stop_words="english")`
- Compute pairwise cosine similarity
- Return dict with:
  - `similarity_matrix`: 2D array (n×n)
  - `most_similar_pair`: tuple of (file1, file2, similarity_score)
  - `distinctive_terms`: dict mapping filename to top 5 TF-IDF terms

**Investigation note:** The distinctive terms reveal what's unique about each case. If multiple cases share the same distinctive symptom or medication, that's a strong clue. The similarity matrix shows which cases cluster together.

**Test behavior:**

- Function executes without errors
- Returns dict with expected keys
- Similarity matrix is square (n×n) where n = number of files
- Values in [0, 1] range
- Most similar pair has highest similarity score
- Distinctive terms are strings (top TF-IDF terms per document)

**Artifact:** Function that produces comparison results

## Part 4: Build the Case

**File:** `solve_mystery.py`

**Narrative:** Now combine everything into a complete pipeline. Process each case report through your pipeline to build a comprehensive analysis. The pipeline output will help you identify the common thread connecting all cases.

**Task:** Build a complete NLP pipeline combining preprocessing, extraction, and analysis.

**Requirements:**

- Function: `analyze_case(text: str, corpus: list[str]) -> dict`
- Combine steps from Parts 1-3:
  - Preprocess text (tokenize, normalize, lemmatize)
  - Extract entities and patterns (names, dates, BP, medications)
  - Compute TF-IDF vector for the document (use the full corpus for IDF calculation)
- Return dict with: `tokens`, `entities`, `extracted_info`, `tfidf_vector` (sparse array or dense)

**Investigation note:** By analyzing all cases through the same pipeline, you can systematically compare them. The common thread might be in the distinctive terms, the medications, or the timeline of dates.

**Test behavior:**

- Function executes without errors
- Returns dict with expected keys
- Tokens are preprocessed (lemmatized, stopwords removed)
- Entities extracted (spaCy NER)
- TF-IDF vector computed (can be sparse or dense array)

**Artifact:** Complete pipeline function

**Optional bonus:** After implementing all parts, students can write a short analysis (in comments or print statements) identifying what connects the cases based on the pipeline outputs.

## Test Suite Design

Following `lectures/01/assignment/.github/tests/test_assignment.py` pattern:

**Test behaviors, not implementation:**

- Import functions and test with known inputs
- Verify output structure (dict keys, list types)
- Check known input/output pairs
- Verify artifacts (functions exist, execute without errors)
- Do NOT check: function names, variable names, code structure, specific imports

**Test classes:**

1. `TestPart1Preprocessing` - function execution, output format, known I/O pairs
2. `TestPart2Extraction` - function execution, dict structure, entity extraction
3. `TestPart3Comparison` - function execution, similarity matrix format, distinctive terms
4. `TestPart4Pipeline` - function execution, combined output structure

## README.md Structure

Following `lectures/01/assignment/README.md` pattern but with investigation theme:

- **Opening** - "The Symptom Cluster Investigation" narrative setup
  - Brief: Multiple patients, similar symptoms, need to find the connection
  - Your role: Use NLP to analyze free-text reports and solve the mystery
- **Overview** - pass/fail, autograded, four parts
- **Assignment Structure** - file listing
- **The Investigation** - narrative intro to the cases
- **Part 1-4** sections with:
  - Investigation framing (what this step reveals)
  - File name
  - Your task
  - Requirements (function signature, expected behavior)
  - Investigation note (what to look for)
  - Success criteria checklist
- **Testing Your Work** - how to run pytest locally
- **Submission Checklist**
- **Grading** - pass/fail criteria table
- **Getting Help** - reference to lecture/demos
- **The Solution** (optional, in hints.md or separate file) - what connects the cases

## Implementation Notes

- Use synthetic clinical-style text (not real patient data) written in engaging narrative style
- Each case report should feel like a real clinical note but with story elements
- The "mystery" should be solvable by analyzing the outputs (e.g., all patients took Drug X, all visited Location Y, all have Symptom Z)
- Provide starter code with function signatures and docstrings
- Include sample data files in `data/` directory
- Tests use fixtures (sample text files) for known input/output validation
- All functions should handle edge cases gracefully (empty text, no entities found, etc.)
- Config-driven optional (not required, but can mention in hints)
- Consider adding a "solution" reveal in hints.md or as a comment in the final file showing what connects the cases

## Dependencies

- `requirements.txt`: nltk, spacy, scikit-learn, pyyaml
- Test requirements: pytest, jupyter, nbconvert (for any notebook parts if added)
- spaCy model: `en_core_web_sm` (can be in requirements.txt as wheel URL like demos)

## Success Criteria

Students who understand the lecture should be able to:

1. Implement preprocessing using NLTK tokenization and lemmatization
2. Use spaCy for POS and NER extraction
3. Write regex patterns for structured data (BP, medications, dates)
4. Vectorize documents with TF-IDF and compute similarity
5. Combine steps into a reusable pipeline function

All parts are testable via function imports and known input/output pairs.