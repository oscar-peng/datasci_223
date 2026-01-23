Natural Language Processing

# Links & Self-Guided Review

- [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) — Jurafsky & Martin, comprehensive NLP textbook (free online, chapters 2-6 cover this lecture)
- [NLTK Book](https://www.nltk.org/book/) — official tutorial with exercises (chapters 1-7)
- [spaCy 101](https://spacy.io/usage/spacy-101) — core concepts and usage patterns
- [Real Python: NLP with spaCy](https://realpython.com/natural-language-processing-spacy-python/) — hands-on tutorial
- [Regex101](https://regex101.com/) — interactive regex tester with explanation
- [scispaCy](https://allenai.github.io/scispacy/) — biomedical NLP models

---

# Natural Language Processing

## What is NLP?

Humans communicate in natural language—English, Spanish, clinical shorthand. Computers need structure—numbers, categories, defined relationships. Natural language processing (NLP) bridges this gap, transforming free-form text into data that algorithms can analyze.

NLP powers everyday tools:

- **Search engines** understand queries and match relevant documents
- **Translation services** convert between languages
- **Voice assistants** interpret spoken commands
- **Email filters** detect spam and categorize messages

The core challenges:

- **Ambiguity** — "bank" means river bank or financial bank?
- **Context** — "not bad" means good
- **Variation** — "BP", "blood pressure", "b.p." all mean the same thing
- **Implicit knowledge** — "take with food" implies meals

## Why NLP for Health Data?

Electronic health records contain vast amounts of free-text data: physician notes, discharge summaries, radiology reports, pathology findings. Surveys and patient-reported outcomes add more unstructured text. NLP lets you:

- **Extract diagnoses** from clinical notes that weren't coded
- **Identify adverse events** mentioned in free text
- **Analyze sentiment** (positive/negative tone) in patient feedback
- **Build cohorts** from text descriptions that predate structured fields

| Data Source | Example Text | What NLP Can Extract |
|-------------|--------------|----------------------|
| Progress notes | "Patient denies chest pain, reports mild fatigue" | Symptoms (negated and affirmed) |
| Radiology reports | "No acute intracranial abnormality" | Findings, negation status |
| Discharge summaries | "Follow up with cardiology in 2 weeks" | Care instructions, timing |
| Patient surveys | "The wait time was frustrating" | Sentiment, specific complaints |

## Tools: NLTK vs spaCy

Two Python libraries dominate classical NLP work.

**NLTK (Natural Language Toolkit)** is designed for learning and research. It offers many algorithms for each task, letting you explore different approaches. Processing is string-based—you work with lists of words and manual pipelines.

**spaCy** is designed for production applications. It provides one optimized algorithm per task, prioritizing speed and ease of use. Processing is object-oriented—you work with `Doc`, `Token`, and `Span` objects that carry rich annotations.

| Aspect | NLTK | spaCy |
|--------|------|-------|
| Philosophy | Educational, comprehensive | Production-ready, fast |
| Algorithm choice | Many algorithms to choose | One best algorithm per task |
| Processing style | String-based | Object-oriented (Doc, Token, Span) |
| Pipeline | Manual assembly | Integrated pipeline |
| Best for | Learning, research | Applications |

**spaCy's object model:**

- **Doc** — a processed document containing all tokens and annotations
- **Token** — a single word or punctuation mark with attributes (text, POS, lemma)
- **Span** — a slice of a Doc (like a substring, but with token information)

### Reference Card: Tool Installation

| Tool | Install | First-time Setup |
|------|---------|------------------|
| NLTK | `pip install nltk` | `nltk.download('punkt')`, `nltk.download('stopwords')` |
| spaCy | `pip install spacy` | `python -m spacy download en_core_web_sm` |
| scikit-learn | `pip install scikit-learn` | (none required) |

## Classical vs LLM-based Approaches

This lecture covers **classical NLP**—techniques developed before large language models became practical. These methods remain valuable and widely used.

| Aspect | Classical NLP | LLM-based |
|--------|---------------|-----------|
| Text representation | Word counts, TF-IDF | Contextual embeddings |
| Pipeline | Explicit stages (tokenize → analyze → vectorize) | Often end-to-end |
| Interpretability | High—you can inspect features | Lower—embeddings are opaque |
| Computational cost | Low | High |
| Training data | Works with small labeled sets | Benefits from massive pretraining |

**When to use classical NLP:**

- You need interpretable features ("which words predict readmission?")
- Computational resources are limited
- You're building rule-based extraction
- The task is well-defined and doesn't require deep understanding

**When to use LLMs:**

- The task requires understanding context and nuance
- You need text generation or summarization
- Transfer learning from general knowledge helps your domain

Most real-world clinical NLP systems combine both: classical techniques for structured extraction, LLMs for complex reasoning.

---

# Text Processing Fundamentals

Before analysis, we transform raw text into a consistent format. These preprocessing steps are foundational to nearly all NLP work.

## Tokenization

Tokenization splits text into individual units called **tokens**—usually words, but sometimes punctuation, numbers, or subwords. Every subsequent NLP step operates on tokens.

**Why tokenization matters:**

- "500mg" as one token vs. "500" + "mg" affects downstream analysis
- Abbreviations like "Dr." shouldn't be split at the period
- Medical terms like "COVID-19" should stay together

**Tokenization and NLP methods:** The choice of tokenization strategy depends on your downstream task. Classical NLP typically uses word-level tokenization. Modern LLMs use **subword tokenization** (BPE, WordPiece), which splits words into smaller pieces like "pre" + "process" + "ing"—this handles rare words better and creates a fixed vocabulary size. The tokenizer and model must match: you can't use a word tokenizer with a model trained on subwords.

```
Sentence: "Dr. Smith prescribed 500mg ibuprofen."

Whitespace split:   ["Dr.", "Smith", "prescribed", "500mg", "ibuprofen."]
                    ↑ keeps punctuation attached

NLTK word_tokenize: ["Dr.", "Smith", "prescribed", "500mg", "ibuprofen", "."]
                    ↑ handles abbreviations, splits final "."

spaCy tokenizer:    ["Dr.", "Smith", "prescribed", "500", "mg", "ibuprofen", "."]
                    ↑ splits numbers from units
```

### Reference Card: Tokenization

| Tool | Function | Notes |
|------|----------|-------|
| Python | `text.split()` | Splits on whitespace only |
| NLTK | `nltk.word_tokenize(text)` | Handles punctuation, abbreviations |
| NLTK | `nltk.sent_tokenize(text)` | Splits into sentences |
| spaCy | `for token in nlp(text)` | Tokens via iteration |

### Code Snippet: Tokenization

```python
import nltk
nltk.download('punkt_tab')

text = "Dr. Smith prescribed 500mg ibuprofen. Take twice daily."

# Naive split
print(text.split())
# ['Dr.', 'Smith', 'prescribed', '500mg', 'ibuprofen.', 'Take', 'twice', 'daily.']

# NLTK word tokenize
print(nltk.word_tokenize(text))
# ['Dr.', 'Smith', 'prescribed', '500mg', 'ibuprofen', '.', 'Take', 'twice', 'daily', '.']

# NLTK sentence tokenize
print(nltk.sent_tokenize(text))
# ['Dr. Smith prescribed 500mg ibuprofen.', 'Take twice daily.']
```

## Normalization

Normalization transforms tokens into a consistent form.

**Lowercasing** reduces vocabulary size by treating "Patient" and "patient" as the same word.

> **Caveat:** Lowercasing can destroy useful information. "US" (United States) becomes "us" (pronoun). In clinical text, abbreviations often rely on case: "MS" could mean multiple sclerosis, mental status, or morphine sulfate.

**Stopword removal** filters out common words like "the", "is", "and" that appear frequently but carry little meaning for many tasks. A **stopword list** is a predefined set of these common words.

> **Caveat:** Stopwords aren't always useless. "No chest pain" loses critical meaning if you remove "no". For clinical text, be careful with negation words.

**Punctuation removal** strips commas, periods, and other marks—unless they carry meaning (like hyphens in "COVID-19").

### Reference Card: Normalization

| Operation | NLTK | Python Built-in |
|-----------|------|-----------------|
| Lowercase | `text.lower()` | `text.lower()` |
| Remove punctuation | Use `string.punctuation` | `str.translate()` |
| Stopwords list | `stopwords.words('english')` | — |

### Code Snippet: Normalization

```python
import string
import nltk
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

text = "The Patient, age 45, presents WITH chest pain."

text = text.lower()
text = text.translate(str.maketrans('', '', string.punctuation))
tokens = nltk.word_tokenize(text)

stop_words = set(stopwords.words('english'))
tokens = [t for t in tokens if t not in stop_words]

print(tokens)
# ['patient', 'age', '45', 'presents', 'chest', 'pain']
```

## Stemming and Lemmatization

Both techniques reduce words to a common base form, helping group related words together.

**Stemming** chops off word endings using simple rules. It's fast but crude—"studies" becomes "studi" (not a real word).

**Lemmatization** uses vocabulary and word structure analysis to find the actual dictionary form (the **lemma**). "studies" → "study", "better" → "good". More accurate but slower.

```
Word          Stemmer Output    Lemmatizer Output
───────────────────────────────────────────────────
"running"     "run"             "run"
"studies"     "studi"           "study"
"better"      "better"          "good" (with POS=adj)
"universities""univers"         "university"
```

### Reference Card: Stemming & Lemmatization

| Tool | Class/Function | Notes |
|------|----------------|-------|
| NLTK | `PorterStemmer()` | Classic English stemmer |
| NLTK | `SnowballStemmer('english')` | Improved Porter variant |
| NLTK | `WordNetLemmatizer()` | Requires POS for best results |
| spaCy | `token.lemma_` | Built into pipeline |

### Code Snippet: Stemming vs Lemmatization

```python
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
nltk.download('wordnet')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words = ["running", "studies", "better", "caring"]

for word in words:
    print(f"{word}: stem={stemmer.stem(word)}, lemma={lemmatizer.lemmatize(word, pos='v')}")
```

---

# Text Representation

To use text in machine learning, we need numerical representations. These classical approaches convert documents into vectors (lists of numbers).

## Bag of Words

**Bag of Words (BoW)** counts how many times each word appears, ignoring order. The result is a **document-term matrix** where each row is a document and each column is a word from the **vocabulary** (all unique words across documents).

![Document-Term Matrix Heatmap](media/bow_heatmap.png)

```
Documents:
1. "patient reports chest pain"
2. "patient denies chest pain"
3. "patient reports headache"

Vocabulary: [chest, denies, headache, pain, patient, reports]

Document-Term Matrix:
           chest  denies  headache  pain  patient  reports
Doc 1        1       0        0       1       1        1
Doc 2        1       1        0       1       1        0
Doc 3        0       0        1       0       1        1
```

**Limitations:**

- Ignores word order ("patient reports pain" = "pain reports patient")
- Creates **sparse matrices** (most values are 0)
- Common words dominate the counts

### Reference Card: Bag of Words

| Tool | Class | Key Parameters |
|------|-------|----------------|
| scikit-learn | `CountVectorizer()` | `max_features`, `stop_words`, `ngram_range` |
| Method | `.fit_transform(docs)` | Returns sparse matrix |
| Method | `.get_feature_names_out()` | Returns vocabulary |
| Method | `.toarray()` | Convert sparse to dense |

### Code Snippet: Bag of Words

```python
from sklearn.feature_extraction.text import CountVectorizer

docs = [
    "patient reports chest pain",
    "patient denies chest pain",
    "patient reports headache"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)

print(vectorizer.get_feature_names_out())
# ['chest' 'denies' 'headache' 'pain' 'patient' 'reports']

print(X.toarray())
# [[1 0 0 1 1 1]
#  [1 1 0 1 1 0]
#  [0 0 1 0 1 1]]
```

## TF-IDF

**TF-IDF (Term Frequency–Inverse Document Frequency)** improves on raw counts by weighting words based on how distinctive they are. Words that appear in every document get downweighted; rare, specific terms get upweighted.

![TF-IDF Weights](media/tfidf_weights.png)

$$\text{TF-IDF}(word, doc) = \text{TF}(word, doc) \times \text{IDF}(word)$$

- **TF (Term Frequency)** — how often the word appears in this document
- **IDF (Inverse Document Frequency)** — how rare the word is across all documents: $\log(\text{total docs} / \text{docs containing word})$

**Example:** "diabetes" appears in 10 of 1000 documents → IDF ≈ 2.0 (distinctive). "patient" appears in 900 of 1000 → IDF ≈ 0.05 (common).

### Reference Card: TF-IDF

| Tool | Class | Notes |
|------|-------|-------|
| scikit-learn | `TfidfVectorizer()` | Combines tokenization + TF-IDF |
| scikit-learn | `TfidfTransformer()` | Applies to existing count matrix |
| Parameters | `max_df`, `min_df` | Filter by document frequency |
| Parameters | `ngram_range` | Include word pairs/triples |

### Code Snippet: TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer

docs = [
    "patient reports chest pain",
    "patient denies chest pain",
    "patient reports headache"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)

for word, idf in zip(vectorizer.get_feature_names_out(), vectorizer.idf_):
    print(f"{word}: IDF = {idf:.2f}")

# patient: IDF = 1.00   ← appears in all docs, lowest IDF
# denies: IDF = 1.69    ← appears in only 1 doc, high IDF
```

## N-grams

Single words (**unigrams**) lose context. **N-grams** capture sequences of N consecutive words.

- **Bigrams** (n=2): word pairs — "chest pain", "denies chest"
- **Trigrams** (n=3): word triples — "patient denies chest"

**Why n-grams matter for clinical text:**

- "chest pain" is meaningful as a unit
- "denies chest pain" captures negation context
- "no chest pain" vs "chest pain" mean opposite things

### Reference Card: N-grams

| Tool | Parameter | Effect |
|------|-----------|--------|
| `CountVectorizer` | `ngram_range=(1, 1)` | Unigrams only (default) |
| `CountVectorizer` | `ngram_range=(1, 2)` | Unigrams and bigrams |
| `TfidfVectorizer` | `ngram_range=(1, 3)` | Unigrams through trigrams |

### Code Snippet: N-grams

```python
from sklearn.feature_extraction.text import CountVectorizer

docs = ["patient denies chest pain", "patient reports chest pain"]

vectorizer = CountVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(docs)

print(vectorizer.get_feature_names_out())
# ['chest', 'chest pain', 'denies', 'denies chest', 'pain',
#  'patient', 'patient denies', 'patient reports', 'reports', 'reports chest']
```

## Word Vectors

The representations above treat each word independently—"diabetes" and "hypertension" are just as different as "diabetes" and "pizza." **Word vectors** (embeddings) capture semantic similarity: related words have similar vectors.

spaCy's medium and large models (`en_core_web_md`, `en_core_web_lg`) include pre-computed word vectors. We'll cover how embeddings work in Lecture 07.

For classical NLP, TF-IDF is your go-to representation: interpretable, effective, and doesn't require special models.

![XKCD: Spelling](media/xkcd_spelling.png)

---

# LIVE DEMO!

---

# Part-of-Speech Tagging

Part-of-speech (POS) tagging labels each token with its grammatical role: noun, verb, adjective, etc.

## Concepts

POS tagging enables:

- **Better lemmatization** — knowing "running" is a verb vs noun
- **Information extraction** — find all nouns to identify topics
- **Syntax analysis** — understand sentence structure

```
"The patient reported severe chest pain yesterday."

The      → DT  (determiner)
patient  → NN  (noun, singular)
reported → VBD (verb, past tense)
severe   → JJ  (adjective)
chest    → NN  (noun, singular)
pain     → NN  (noun, singular)
```

Tags follow standardized sets. **Penn Treebank** tags (NLTK) are the traditional English standard. **Universal Dependencies** tags (spaCy) work across languages.

### Common POS Tags

| Tag | Description | Example |
|-----|-------------|---------|
| NN | Noun, singular | patient, pain |
| NNS | Noun, plural | patients, symptoms |
| VB | Verb, base form | diagnose, treat |
| VBD | Verb, past tense | diagnosed, treated |
| VBG | Verb, gerund | diagnosing, running |
| JJ | Adjective | severe, chronic |
| RB | Adverb | quickly, very |
| DT | Determiner | the, a, an |

## NLTK

### Reference Card: NLTK POS Tagging

| Function | Purpose | Returns |
|----------|---------|---------|
| `nltk.pos_tag(tokens)` | Tag tokenized text | List of (word, tag) tuples |
| `nltk.help.upenn_tagset('NN')` | Explain a tag | Tag description |

### Code Snippet: NLTK POS Tagging

```python
import nltk
nltk.download('averaged_perceptron_tagger_eng')

text = "The patient reported severe chest pain."
tokens = nltk.word_tokenize(text)
tagged = nltk.pos_tag(tokens)

print(tagged)
# [('The', 'DT'), ('patient', 'NN'), ('reported', 'VBD'),
#  ('severe', 'JJ'), ('chest', 'NN'), ('pain', 'NN'), ('.', '.')]

# Find all nouns
nouns = [word for word, tag in tagged if tag.startswith('NN')]
print(nouns)  # ['patient', 'chest', 'pain']
```

## spaCy

### Reference Card: spaCy POS Tagging

| Attribute | Purpose | Tag Set |
|-----------|---------|---------|
| `token.pos_` | Coarse POS tag | Universal Dependencies |
| `token.tag_` | Fine-grained tag | Penn Treebank style |
| `spacy.explain(tag)` | Explain a tag | Description string |

### Code Snippet: spaCy POS Tagging

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The patient reported severe chest pain.")

for token in doc:
    print(f"{token.text:12} {token.pos_:6} {token.tag_}")

# The          DET    DT
# patient      NOUN   NN
# reported     VERB   VBD
# severe       ADJ    JJ
# chest        NOUN   NN
# pain         NOUN   NN
```

![XKCD: Language Acquisition](media/xkcd_language_acquisition.png)

---

# Named Entity Recognition

Named Entity Recognition (NER) identifies and classifies specific entities in text: people, organizations, locations, dates. For clinical text, specialized models can extract medications, dosages, diagnoses, and procedures.

## Concepts

<!-- #FIXME: Replace with image - search: "named entity recognition NER clinical text visualization highlighted entities" -->

**Example:** "Dr. Smith at UCSF prescribed Metformin 500mg on January 15."

| Text | Entity Type |
|------|-------------|
| Dr. Smith | PERSON |
| UCSF | ORG (organization) |
| Metformin | (needs medical NER model) |
| 500mg | QUANTITY |
| January 15 | DATE |

**Standard NER entities:** PERSON, ORG, GPE (location), DATE, TIME, MONEY, PERCENT

**Clinical NER entities** (specialized models): MEDICATION, DOSAGE, DIAGNOSIS, PROCEDURE, ANATOMY

## NLTK

NLTK's NER requires POS-tagged input and returns a tree structure.

### Reference Card: NLTK NER

| Function | Purpose | Returns |
|----------|---------|---------|
| `nltk.ne_chunk(tagged)` | Extract entities | Tree structure |
| `nltk.ne_chunk(tagged, binary=True)` | Just NE vs not | Tree with NE labels only |

### Code Snippet: NLTK NER

```python
import nltk
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')

text = "Dr. Smith at UCSF prescribed medication."
tokens = nltk.word_tokenize(text)
tagged = nltk.pos_tag(tokens)
entities = nltk.ne_chunk(tagged)

for chunk in entities:
    if hasattr(chunk, 'label'):
        print(f"{' '.join(c[0] for c in chunk)}: {chunk.label()}")
# UCSF: ORGANIZATION
```

## spaCy

### Reference Card: spaCy NER

| Attribute | Purpose | Returns |
|-----------|---------|---------|
| `doc.ents` | All entities | Tuple of Span objects |
| `ent.text` | Entity text | String |
| `ent.label_` | Entity type | String (PERSON, ORG, etc.) |
| `ent.start`, `ent.end` | Token positions | Integers |

### Code Snippet: spaCy NER

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Dr. Smith at UCSF prescribed medication on January 15.")

for ent in doc.ents:
    print(f"{ent.text}: {ent.label_}")

# Dr. Smith: PERSON
# UCSF: ORG
# January 15: DATE
```

---

# Text Extraction

Beyond NER, we often need to extract specific patterns from text—vitals, dosages, dates, and other structured information.

## Regex Patterns

Regular expressions (regex) are pattern-matching tools. Where NER identifies semantic entities, regex extracts syntactic patterns.

```
Pattern     Matches                Example
────────────────────────────────────────────────────────
\d          digit                  "5" in "500mg"
\d+         one or more digits     "500" in "500mg"
\w+         word characters        "patient" in "patient:"
[A-Z]+      uppercase letters      "BP" in "BP: 120/80"
\s          whitespace             spaces, tabs, newlines
(...)       capture group          extract matched portion
|           OR                     "mg|ml|mcg"

Clinical patterns:
• Vitals:  \d{2,3}/\d{2,3}        →  "120/80"
• Dosage:  \d+\s?(mg|ml|mcg)      →  "500 mg", "10ml"
• Date:    \d{1,2}/\d{1,2}/\d{4}  →  "01/15/2025"
```

### Reference Card: Python Regex

| Function | Purpose | Returns |
|----------|---------|---------|
| `re.search(pattern, text)` | Find first match | Match object or None |
| `re.findall(pattern, text)` | Find all matches | List of strings |
| `re.sub(pattern, repl, text)` | Replace matches | Modified string |
| `match.group()` | Get matched text | String |
| `match.groups()` | Get capture groups | Tuple |

### Code Snippet: Clinical Text Extraction

```python
import re

note = """
Patient vitals: BP 120/80, HR 72, Temp 98.6F
Medications: Metformin 500mg twice daily, Lisinopril 10mg daily
Lab results from 01/15/2025: HbA1c 7.2%
"""

# Extract blood pressure
bp_readings = re.findall(r'\d{2,3}/\d{2,3}', note)
print(f"BP: {bp_readings}")  # ['120/80']

# Extract medication dosages
medications = re.findall(r'(\w+)\s+(\d+)\s?(mg|ml)', note)
print(f"Meds: {medications}")  # [('Metformin', '500', 'mg'), ('Lisinopril', '10', 'mg')]

# Extract dates
dates = re.findall(r'\d{1,2}/\d{1,2}/\d{4}', note)
print(f"Dates: {dates}")  # ['01/15/2025']
```

![XKCD: Regex Golf](media/xkcd_regex_golf.png)

---

# LIVE DEMO!!

---

# Document Similarity

With text represented as vectors, we can measure how similar documents are. This enables search, clustering, and recommendation systems.

## Cosine Similarity

**Cosine similarity** measures the angle between two vectors rather than their distance. This makes it robust to document length—a long document and a short document about the same topic will have high similarity.

![Document Similarity in Vector Space](media/document_similarity.png)

$$\cos(\theta) = \frac{A \cdot B}{\|A\| \times \|B\|}$$

- **Range:** 0 to 1 for TF-IDF vectors
- **1.0** = identical direction (very similar)
- **0.0** = perpendicular (no shared words)

### Reference Card: Document Similarity

| Tool | Function | Returns |
|------|----------|---------|
| scikit-learn | `cosine_similarity(X)` | Pairwise similarity matrix |
| scikit-learn | `cosine_similarity(X, Y)` | Similarity between X and Y |
| scipy | `spatial.distance.cosine(u, v)` | Cosine distance (1 - similarity) |

### Code Snippet: Document Similarity

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

docs = [
    "patient presents with chest pain and shortness of breath",
    "patient reports chest discomfort and difficulty breathing",
    "patient complains of headache and nausea"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)

similarities = cosine_similarity(X)
print(similarities)
# [[1.   0.35 0.11]
#  [0.35 1.   0.11]
#  [0.11 0.11 1.  ]]
# Docs 0 and 1 are most similar (both about chest/breathing)
```

---

# Pipelines

## spaCy: Integrated Processing

spaCy processes text through a unified pipeline—tokenization, POS tagging, lemmatization, and NER in one pass.

### Reference Card: spaCy Pipeline

| Component | Access | Description |
|-----------|--------|-------------|
| `nlp = spacy.load("en_core_web_sm")` | Load model | Small English model |
| `doc = nlp(text)` | Process text | Returns Doc object |
| `token.text` | Original text | String |
| `token.lower_` | Lowercased | String |
| `token.lemma_` | Base form | String |
| `token.pos_` | POS tag | Universal Dependencies |
| `token.dep_` | Dependency | Syntactic role |
| `doc.ents` | Entities | Tuple of Spans |

### Code Snippet: spaCy Pipeline

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The patient was diagnosed with Type 2 diabetes.")

for token in doc:
    print(f"{token.text:12} {token.pos_:6} {token.lemma_:12} {token.dep_}")

# The          DET    the          det
# patient      NOUN   patient      nsubjpass
# was          AUX    be           auxpass
# diagnosed    VERB   diagnose     ROOT
# with         ADP    with         prep
# Type         PROPN  Type         compound
# 2            NUM    2            compound
# diabetes     NOUN   diabetes     pobj
```

![XKCD: Python Environment](media/xkcd_python_environment.png)

## Combining spaCy + Regex

```python
import spacy
import re

def process_note(note):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(note)

    return {
        'entities': [(ent.text, ent.label_) for ent in doc.ents],
        'blood_pressure': re.findall(r'\d{2,3}/\d{2,3}', note),
        'nouns': [token.lemma_ for token in doc if token.pos_ == "NOUN"]
    }

note = "Patient John Smith, age 45, presents with BP 140/90 and chest pain."
print(process_note(note))
# {'entities': [('John Smith', 'PERSON'), ('45', 'DATE')],
#  'blood_pressure': ['140/90'],
#  'nouns': ['patient', 'age', 'chest', 'pain']}
```

## Challenges

General NLP tools struggle with clinical text:

**Abbreviations** — "pt c/o SOB" = "patient complains of shortness of breath". Same abbreviation, different meanings: "MS" = multiple sclerosis OR mental status OR morphine sulfate.

**Negation** — "Patient denies chest pain" means chest pain is ABSENT. Simple keyword extraction misses this.

**Uncertainty** — "possible pneumonia" ≠ "confirmed pneumonia". "rule out MI" = suspicion, not diagnosis.

**Temporality** — "History of diabetes" (past) vs "Patient has diabetes" (current) vs "Risk of diabetes" (future).

**Variations** — "hypertention", "htn", "HTN", "high blood pressure" all mean the same thing.

## Specialized Tools

| Tool | Focus | Access |
|------|-------|--------|
| scispaCy | Biomedical NER | `pip install scispacy` |
| MedSpaCy | Negation detection, clinical pipelines | `pip install medspacy` |
| cTAKES | Full clinical NLP (Java) | Apache, open source |
| MetaMap | UMLS concept extraction | NLM, requires license |

**UMLS** (Unified Medical Language System) — biomedical vocabulary database from the National Library of Medicine.

---

# LIVE DEMO!!!

![XKCD: Machine Learning](media/xkcd_machine_learning.png)
