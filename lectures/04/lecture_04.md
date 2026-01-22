Natural Language Processing

# Overview

Natural language processing (NLP) transforms unstructured text—clinical notes, research abstracts, patient surveys—into structured, analyzable data. In health informatics, this unlocks insights from the massive volume of text that clinicians and researchers generate daily.

This lecture covers foundational NLP techniques you can apply immediately:

- **Text preprocessing**: cleaning, tokenization, normalization
- **Linguistic analysis**: part-of-speech tagging, named entity recognition
- **Text representation**: bag-of-words, TF-IDF
- **Practical tools**: NLTK for learning fundamentals, spaCy for production work

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         NLP Pipeline Overview                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Raw Text ──► Preprocessing ──► Tokenization ──► Analysis ──► Output  │
│                                                                         │
│   "Patient     lowercase,        ["patient",      POS tags,    structured│
│    reports     remove noise       "reports",      entities,    data,     │
│    mild..."                       "mild", ...]    vectors      features  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

# Why NLP for Health Data?

Electronic health records contain vast amounts of free-text data: physician notes, discharge summaries, radiology reports, pathology findings. Surveys and patient-reported outcomes add more unstructured text. NLP lets you:

- **Extract diagnoses** from clinical notes that weren't coded
- **Identify adverse events** mentioned in free text
- **Analyze sentiment** in patient feedback
- **Build cohorts** from text descriptions that predate structured fields

| Data Source | Example Text | What NLP Can Extract |
|-------------|--------------|----------------------|
| Progress notes | "Patient denies chest pain, reports mild fatigue" | Symptoms (negated and affirmed) |
| Radiology reports | "No acute intracranial abnormality" | Findings, negation status |
| Discharge summaries | "Follow up with cardiology in 2 weeks" | Care instructions, timing |
| Patient surveys | "The wait time was frustrating" | Sentiment, specific complaints |

---

# Text Preprocessing

Raw text is messy. Before analysis, you'll typically:

1. **Lowercase** – reduces vocabulary size ("Patient" = "patient")
2. **Remove punctuation** – unless it carries meaning (negation, abbreviations)
3. **Remove stopwords** – common words like "the", "is", "and" that add little meaning
4. **Handle whitespace** – normalize spaces, remove extra newlines

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Preprocessing Pipeline                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   "The Patient, age 45, presents WITH chest pain."                      │
│                           │                                             │
│                           ▼                                             │
│   ┌─────────────────────────────────────────────┐                       │
│   │  1. Lowercase                               │                       │
│   │  "the patient, age 45, presents with..."    │                       │
│   └─────────────────────────────────────────────┘                       │
│                           │                                             │
│                           ▼                                             │
│   ┌─────────────────────────────────────────────┐                       │
│   │  2. Remove punctuation                      │                       │
│   │  "the patient age 45 presents with..."      │                       │
│   └─────────────────────────────────────────────┘                       │
│                           │                                             │
│                           ▼                                             │
│   ┌─────────────────────────────────────────────┐                       │
│   │  3. Remove stopwords                        │                       │
│   │  "patient age 45 presents chest pain"       │                       │
│   └─────────────────────────────────────────────┘                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Reference Card: Basic Preprocessing

| Operation | NLTK | Python Built-in |
|-----------|------|-----------------|
| Lowercase | `text.lower()` | `text.lower()` |
| Remove punctuation | Use `string.punctuation` | `str.translate()` |
| Stopwords list | `nltk.corpus.stopwords.words('english')` | — |
| Word tokenize | `nltk.word_tokenize(text)` | `text.split()` (naive) |

### Code Snippet: Basic Preprocessing

```python
import string
import nltk
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

text = "The Patient, age 45, presents WITH chest pain."

# Lowercase
text = text.lower()

# Remove punctuation
text = text.translate(str.maketrans('', '', string.punctuation))

# Tokenize
tokens = nltk.word_tokenize(text)

# Remove stopwords
stop_words = set(stopwords.words('english'))
tokens = [t for t in tokens if t not in stop_words]

print(tokens)
# ['patient', 'age', '45', 'presents', 'chest', 'pain']
```

---

# Tokenization

Tokenization splits text into individual units (tokens)—usually words, but sometimes subwords or characters. This is the foundation of nearly all NLP work.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Tokenization Strategies                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Sentence: "Dr. Smith prescribed 500mg ibuprofen."                     │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │ Whitespace split:  ["Dr.", "Smith", "prescribed", "500mg",     │   │
│   │                     "ibuprofen."]                               │   │
│   │                     ↑ keeps punctuation attached                │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │ NLTK word_tokenize: ["Dr.", "Smith", "prescribed", "500mg",    │   │
│   │                      "ibuprofen", "."]                          │   │
│   │                      ↑ handles abbreviations, splits final "." │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │ spaCy tokenizer:    ["Dr.", "Smith", "prescribed", "500", "mg",│   │
│   │                      "ibuprofen", "."]                          │   │
│   │                      ↑ splits numbers from units                │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Why does tokenization matter?**

- "500mg" as one token vs. "500" + "mg" affects downstream analysis
- Abbreviations like "Dr." shouldn't be split at the period
- Medical terms like "COVID-19" should stay together

### Reference Card: Tokenization

| Tool | Function | Notes |
|------|----------|-------|
| Python | `text.split()` | Splits on whitespace only |
| NLTK | `nltk.word_tokenize(text)` | Handles punctuation, abbreviations |
| NLTK | `nltk.sent_tokenize(text)` | Splits into sentences |
| spaCy | `nlp(text)` returns `Doc` | Tokens accessible via iteration |

### Code Snippet: Tokenization Comparison

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

![XKCD: Spelling](media/xkcd_spelling.png)

---

# Stemming and Lemmatization

Both reduce words to a common base form, but they work differently:

**Stemming** chops off word endings using rules. Fast but crude—"studies" becomes "studi", not "study".

**Lemmatization** uses vocabulary and morphological analysis to find the dictionary form. "studies" → "study", "better" → "good".

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Stemming vs. Lemmatization                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Word          │  Stemmer Output    │  Lemmatizer Output              │
│   ─────────────────────────────────────────────────────────────────     │
│   "running"     │  "run"             │  "run"                          │
│   "studies"     │  "studi"           │  "study"                        │
│   "better"      │  "better"          │  "good" (with POS=adj)          │
│   "caring"      │  "care"            │  "care"                         │
│   "universities"│  "univers"         │  "university"                   │
│                                                                         │
│   Speed:        │  Fast (rules only) │  Slower (dictionary lookup)     │
│   Accuracy:     │  Lower             │  Higher                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Reference Card: Stemming & Lemmatization

| Tool | Class/Function | Notes |
|------|----------------|-------|
| NLTK | `PorterStemmer()` | Classic English stemmer |
| NLTK | `SnowballStemmer('english')` | Improved Porter variant |
| NLTK | `WordNetLemmatizer()` | Requires POS for best results |
| spaCy | `token.lemma_` | Built into pipeline |

### Code Snippet: Stemming vs. Lemmatization

```python
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
nltk.download('wordnet')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words = ["running", "studies", "better", "caring"]

for word in words:
    print(f"{word}: stem={stemmer.stem(word)}, lemma={lemmatizer.lemmatize(word, pos='v')}")

# running: stem=run, lemma=run
# studies: stem=studi, lemma=study
# better: stem=better, lemma=better
# caring: stem=care, lemma=care
```

---

# LIVE DEMO!

---

# Part-of-Speech Tagging

Part-of-speech (POS) tagging labels each word with its grammatical role: noun, verb, adjective, etc. This enables:

- **Better lemmatization** (knowing "running" is a verb vs. noun)
- **Information extraction** (find all nouns to identify topics)
- **Syntax analysis** (understand sentence structure)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        POS Tag Examples                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   "The patient reported severe chest pain yesterday."                   │
│                                                                         │
│    The      → DT  (determiner)                                          │
│    patient  → NN  (noun, singular)                                      │
│    reported → VBD (verb, past tense)                                    │
│    severe   → JJ  (adjective)                                           │
│    chest    → NN  (noun, singular)                                      │
│    pain     → NN  (noun, singular)                                      │
│    yesterday→ NN  (noun, singular)                                      │
│    .        → .   (punctuation)                                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Common POS Tags (Penn Treebank)

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
| IN | Preposition | in, on, with |
| PRP | Personal pronoun | he, she, they |

### Reference Card: POS Tagging

| Tool | Function | Tag Set |
|------|----------|---------|
| NLTK | `nltk.pos_tag(tokens)` | Penn Treebank |
| spaCy | `token.pos_` | Universal Dependencies |
| spaCy | `token.tag_` | Fine-grained tags |

### Code Snippet: POS Tagging

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
print(nouns)
# ['patient', 'chest', 'pain']
```

![XKCD: Language Acquisition](media/xkcd_language_acquisition.png)

---

# Named Entity Recognition

Named Entity Recognition (NER) identifies and classifies specific entities in text: people, organizations, locations, dates, medical terms. For clinical text, this might include medications, dosages, diagnoses, and procedures.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Named Entity Recognition                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   "Dr. Smith at UCSF prescribed Metformin 500mg on January 15."         │
│                                                                         │
│    Dr. Smith       → PERSON                                             │
│    UCSF            → ORG (organization)                                 │
│    Metformin       → (would need medical NER model)                     │
│    500mg           → QUANTITY                                           │
│    January 15      → DATE                                               │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  Standard NER entities:                                         │   │
│   │  PERSON, ORG, GPE (location), DATE, TIME, MONEY, PERCENT        │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  Clinical NER entities (specialized models):                    │   │
│   │  MEDICATION, DOSAGE, DIAGNOSIS, PROCEDURE, ANATOMY              │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Reference Card: Named Entity Recognition

| Tool | Function | Notes |
|------|----------|-------|
| NLTK | `nltk.ne_chunk(tagged)` | Returns tree structure |
| spaCy | `doc.ents` | Returns `Span` objects |
| spaCy | `ent.text`, `ent.label_` | Entity text and type |

### Code Snippet: NER with spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
text = "Dr. Smith at UCSF prescribed medication on January 15."
doc = nlp(text)

for ent in doc.ents:
    print(f"{ent.text}: {ent.label_}")

# Dr. Smith: PERSON
# UCSF: ORG
# January 15: DATE
```

---

# spaCy vs. NLTK

Both are essential NLP libraries, but they serve different purposes:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        spaCy vs. NLTK Comparison                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   NLTK                          │  spaCy                                │
│   ──────────────────────────────┼────────────────────────────────────   │
│   Educational, comprehensive    │  Production-ready, fast               │
│   Many algorithms to choose     │  One best algorithm per task          │
│   String-based processing       │  Object-oriented (Doc, Token, Span)   │
│   Manual pipeline assembly      │  Integrated pipeline                  │
│   Good for learning/research    │  Good for applications                │
│   Includes corpora & datasets   │  Focused on processing                │
│                                                                         │
│   Use NLTK when:                │  Use spaCy when:                      │
│   • Learning NLP concepts       │  • Building applications              │
│   • Need specific algorithms    │  • Processing large volumes           │
│   • Academic research           │  • Need speed and efficiency          │
│   • Exploring different methods │  • Want batteries-included            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Reference Card: spaCy Basics

| Component | Access | Description |
|-----------|--------|-------------|
| Load model | `nlp = spacy.load("en_core_web_sm")` | Small English model |
| Process text | `doc = nlp(text)` | Returns `Doc` object |
| Tokens | `for token in doc:` | Iterate through tokens |
| Token text | `token.text` | Original text |
| Lemma | `token.lemma_` | Base form |
| POS tag | `token.pos_` | Universal POS tag |
| Dependency | `token.dep_` | Syntactic dependency |
| Entities | `doc.ents` | Named entities |

### Code Snippet: spaCy Pipeline

```python
import spacy

nlp = spacy.load("en_core_web_sm")
text = "The patient was diagnosed with Type 2 diabetes."
doc = nlp(text)

for token in doc:
    print(f"{token.text:12} {token.pos_:6} {token.lemma_:12} {token.dep_}")

# The          DET    the          det
# patient      NOUN   patient      nsubjpass
# was          AUX    be           auxpass
# diagnosed   VERB   diagnose     ROOT
# with         ADP    with         prep
# Type         PROPN  Type         compound
# 2            NUM    2            compound
# diabetes     NOUN   diabetes     pobj
# .            PUNCT  .            punct
```

![XKCD: Python Environment](media/xkcd_python_environment.png)

---

# LIVE DEMO!!

---

# Text Representation: Bag of Words

To use text in machine learning, we need numerical representations. The simplest approach is **Bag of Words (BoW)**: count how many times each word appears, ignoring order.

![Document-Term Matrix Heatmap](media/bow_heatmap.png)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Bag of Words Example                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Documents:                                                            │
│   1. "patient reports chest pain"                                       │
│   2. "patient denies chest pain"                                        │
│   3. "patient reports headache"                                         │
│                                                                         │
│   Vocabulary: [chest, denies, headache, pain, patient, reports]         │
│                                                                         │
│   Document-Term Matrix:                                                 │
│                                                                         │
│              chest  denies  headache  pain  patient  reports            │
│   Doc 1        1       0        0       1       1        1              │
│   Doc 2        1       1        0       1       1        0              │
│   Doc 3        0       0        1       0       1        1              │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  Limitations:                                                   │   │
│   │  • Ignores word order ("patient reports pain" = "pain reports")│   │
│   │  • Sparse matrices (most values are 0)                         │   │
│   │  • Common words dominate                                       │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Reference Card: Bag of Words

| Tool | Class | Key Parameters |
|------|-------|----------------|
| scikit-learn | `CountVectorizer()` | `max_features`, `stop_words`, `ngram_range` |
| Method | `.fit_transform(docs)` | Returns sparse matrix |
| Method | `.get_feature_names_out()` | Returns vocabulary |

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

---

# TF-IDF: Term Frequency–Inverse Document Frequency

TF-IDF improves on raw counts by weighting words based on how distinctive they are. Words that appear in every document (like "patient") get downweighted; rare, specific terms get upweighted.

![TF-IDF Weights](media/tfidf_weights.png)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           TF-IDF Formula                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   TF-IDF(word, doc) = TF(word, doc) × IDF(word)                         │
│                                                                         │
│   TF (Term Frequency):                                                  │
│   • How often the word appears in this document                         │
│   • TF = count(word in doc) / total words in doc                        │
│                                                                         │
│   IDF (Inverse Document Frequency):                                     │
│   • How rare the word is across all documents                           │
│   • IDF = log(total docs / docs containing word)                        │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  Example:                                                       │   │
│   │  "diabetes" appears in 10 of 1000 documents                     │   │
│   │  IDF = log(1000/10) = log(100) ≈ 2.0  (high—distinctive!)       │   │
│   │                                                                 │   │
│   │  "patient" appears in 900 of 1000 documents                     │   │
│   │  IDF = log(1000/900) ≈ 0.05  (low—common word)                  │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

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

# Show feature names and their IDF values
for word, idf in zip(vectorizer.get_feature_names_out(), vectorizer.idf_):
    print(f"{word}: IDF = {idf:.2f}")

# chest: IDF = 1.29
# denies: IDF = 1.69    ← appears in only 1 doc, high IDF
# headache: IDF = 1.69  ← appears in only 1 doc, high IDF
# pain: IDF = 1.29
# patient: IDF = 1.00   ← appears in all docs, lowest IDF
# reports: IDF = 1.29
```

---

# N-grams: Capturing Word Context

Single words (unigrams) lose context. N-grams capture sequences of N consecutive words, preserving some word order information.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           N-gram Examples                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Text: "patient denies chest pain"                                     │
│                                                                         │
│   Unigrams (n=1): ["patient", "denies", "chest", "pain"]                │
│                                                                         │
│   Bigrams (n=2):  ["patient denies", "denies chest", "chest pain"]      │
│                                                                         │
│   Trigrams (n=3): ["patient denies chest", "denies chest pain"]         │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  Why n-grams matter for clinical text:                          │   │
│   │                                                                 │   │
│   │  • "chest pain" is meaningful as a unit                         │   │
│   │  • "denies chest pain" captures negation context                │   │
│   │  • "no chest pain" vs "chest pain" mean opposite things         │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Code Snippet: N-grams

```python
from sklearn.feature_extraction.text import CountVectorizer

docs = ["patient denies chest pain", "patient reports chest pain"]

# Unigrams + bigrams
vectorizer = CountVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(docs)

print(vectorizer.get_feature_names_out())
# ['chest', 'chest pain', 'denies', 'denies chest', 'pain',
#  'patient', 'patient denies', 'patient reports', 'reports', 'reports chest']
```

![XKCD: Regex Golf](media/xkcd_regex_golf.png)

---

## A Note on Word Vectors

spaCy's medium and large models (`en_core_web_md`, `en_core_web_lg`) include pre-computed word vectors that capture semantic similarity—"diabetes" and "hypertension" are closer together than "diabetes" and "pizza." We'll cover how these work in later lectures. For now, TF-IDF is your go-to text representation: it's interpretable, works well for many tasks, and doesn't require special models.

---

# Document Similarity

With text represented as vectors, we can measure how similar documents are. This enables search, clustering, and recommendation systems.

![Document Similarity in Vector Space](media/document_similarity.png)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Cosine Similarity                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Cosine similarity measures the angle between two vectors:             │
│                                                                         │
│                        A · B                                            │
│   cos(θ) = ─────────────────────                                        │
│             ||A|| × ||B||                                               │
│                                                                         │
│   Range: -1 to 1 (for TF-IDF: 0 to 1 since no negatives)                │
│   • 1.0 = identical direction (very similar)                            │
│   • 0.0 = orthogonal (unrelated)                                        │
│   • -1.0 = opposite (possible with dense vectors)                       │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                     B                                           │   │
│   │                    /                                            │   │
│   │                   /                                             │   │
│   │                  / θ = small angle                              │   │
│   │                 /   → high similarity                           │   │
│   │               A────────────────                                 │   │
│   │                                                                 │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

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

# Compute pairwise similarities
similarities = cosine_similarity(X)
print(similarities)

# [[1.   0.35 0.11]
#  [0.35 1.   0.11]
#  [0.11 0.11 1.  ]]

# Docs 0 and 1 are most similar (both about chest/breathing)
# Doc 2 is different (headache/nausea)
```

---

# LIVE DEMO!!!

---

# Regular Expressions for Text Extraction

Regular expressions (regex) are pattern-matching tools for extracting specific text patterns. Essential for structured extraction from clinical notes.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Common Regex Patterns                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Pattern     │  Matches                │  Example                      │
│   ────────────┼─────────────────────────┼─────────────────────────────  │
│   \d          │  digit                  │  "5" in "500mg"               │
│   \d+         │  one or more digits     │  "500" in "500mg"             │
│   \w+         │  word characters        │  "patient" in "patient:"      │
│   [A-Z]+      │  uppercase letters      │  "BP" in "BP: 120/80"         │
│   .+          │  any characters         │  everything until newline     │
│   \s          │  whitespace             │  spaces, tabs, newlines       │
│   ^           │  start of line          │  "^Diagnosis:"                │
│   $           │  end of line            │  "mg$"                        │
│   (...)       │  capture group          │  extract matched portion      │
│   (?:...)     │  non-capturing group    │  group without extracting     │
│   |           │  OR                     │  "mg|ml|mcg"                  │
│                                                                         │
│   Clinical patterns:                                                    │
│   • Vitals:    \d{2,3}/\d{2,3}         →  "120/80"                      │
│   • Dosage:    \d+\s?(mg|ml|mcg)       →  "500 mg", "10ml"              │
│   • Date:      \d{1,2}/\d{1,2}/\d{4}  →  "01/15/2025"                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Reference Card: Python Regex

| Function | Purpose | Returns |
|----------|---------|---------|
| `re.search(pattern, text)` | Find first match | Match object or None |
| `re.findall(pattern, text)` | Find all matches | List of strings |
| `re.sub(pattern, repl, text)` | Replace matches | Modified string |
| `re.compile(pattern)` | Pre-compile pattern | Regex object |
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

# Extract blood pressure readings
bp_pattern = r'\d{2,3}/\d{2,3}'
bp_readings = re.findall(bp_pattern, note)
print(f"BP: {bp_readings}")  # ['120/80']

# Extract medication dosages
dose_pattern = r'(\w+)\s+(\d+)\s?(mg|ml)'
medications = re.findall(dose_pattern, note)
print(f"Meds: {medications}")  # [('Metformin', '500', 'mg'), ('Lisinopril', '10', 'mg')]

# Extract dates
date_pattern = r'\d{1,2}/\d{1,2}/\d{4}'
dates = re.findall(date_pattern, note)
print(f"Dates: {dates}")  # ['01/15/2025']
```

![XKCD: Automation](media/xkcd_automation.png)

---

# Putting It Together: A Clinical NLP Pipeline

Here's how the pieces fit into a complete pipeline for processing clinical notes:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Clinical NLP Pipeline                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Raw Clinical Note                                                     │
│         │                                                               │
│         ▼                                                               │
│   ┌─────────────────┐                                                   │
│   │ 1. Preprocess   │  lowercase, handle abbreviations                  │
│   └────────┬────────┘                                                   │
│            │                                                            │
│            ▼                                                            │
│   ┌─────────────────┐                                                   │
│   │ 2. Tokenize     │  split into words/sentences                       │
│   └────────┬────────┘                                                   │
│            │                                                            │
│            ▼                                                            │
│   ┌─────────────────┐                                                   │
│   │ 3. NER/POS      │  identify entities, tag grammar                   │
│   └────────┬────────┘                                                   │
│            │                                                            │
│            ▼                                                            │
│   ┌─────────────────┐                                                   │
│   │ 4. Regex        │  extract structured patterns                      │
│   └────────┬────────┘                                                   │
│            │                                                            │
│            ▼                                                            │
│   ┌─────────────────┐                                                   │
│   │ 5. Vectorize    │  TF-IDF for features                              │
│   └────────┬────────┘                                                   │
│            │                                                            │
│            ▼                                                            │
│   Structured Output: features, entities, vectors                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Code Snippet: Simple Clinical Pipeline

```python
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def process_clinical_note(note):
    """Extract structured information from a clinical note."""
    nlp = spacy.load("en_core_web_sm")

    # Process with spaCy
    doc = nlp(note)

    # Extract named entities
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    # Extract vitals with regex
    bp = re.findall(r'\d{2,3}/\d{2,3}', note)

    # Get key nouns (potential symptoms/conditions)
    nouns = [token.lemma_ for token in doc if token.pos_ == "NOUN"]

    return {
        'entities': entities,
        'blood_pressure': bp,
        'key_terms': nouns
    }

note = "Patient John Smith, age 45, presents with BP 140/90 and chest pain."
result = process_clinical_note(note)
print(result)
# {'entities': [('John Smith', 'PERSON'), ('45', 'DATE')],
#  'blood_pressure': ['140/90'],
#  'key_terms': ['patient', 'age', 'chest', 'pain']}
```

---

# Challenges in Clinical NLP

Clinical text has unique challenges that general NLP tools don't handle well out of the box:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Clinical NLP Challenges                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   1. ABBREVIATIONS                                                      │
│      "pt c/o SOB" = "patient complains of shortness of breath"          │
│      Same abbreviation, different meanings: "MS" = multiple sclerosis   │
│                                              OR mental status           │
│                                              OR morphine sulfate        │
│                                                                         │
│   2. NEGATION                                                           │
│      "Patient denies chest pain" → chest pain = ABSENT                  │
│      "No evidence of malignancy" → malignancy = ABSENT                  │
│      Simple keyword extraction misses this!                             │
│                                                                         │
│   3. UNCERTAINTY                                                        │
│      "possible pneumonia" ≠ "confirmed pneumonia"                       │
│      "rule out MI" = suspicion, not diagnosis                           │
│                                                                         │
│   4. TEMPORALITY                                                        │
│      "History of diabetes" = past                                       │
│      "Patient has diabetes" = current                                   │
│      "Risk of diabetes" = future/possible                               │
│                                                                         │
│   5. MISSPELLINGS & VARIATIONS                                          │
│      "hypertention", "htn", "HTN", "high blood pressure"                │
│      All mean the same thing!                                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Specialized Clinical NLP Tools

| Tool | Focus | Access |
|------|-------|--------|
| scispaCy | Biomedical NER | `pip install scispacy` |
| MedSpaCy | Clinical pipelines, negation | `pip install medspacy` |
| cTAKES | Clinical NLP (Java) | Apache, open source |
| MetaMap | UMLS concept extraction | NLM, requires license |

---

# Next Steps

You now have the foundations for working with text data:

- **Preprocessing** cleans and normalizes text
- **Tokenization** splits text into processable units
- **POS tagging** identifies grammatical structure
- **NER** extracts named entities
- **BoW/TF-IDF** creates numerical representations for ML
- **Regex** extracts structured patterns from clinical text

For health data applications, consider:

1. Start with general tools (NLTK, spaCy) to understand your data
2. Move to clinical-specific tools (scispaCy, MedSpaCy) for production
3. Always validate against manual review—NLP isn't perfect
4. Build negation detection into clinical pipelines

![XKCD: Machine Learning](media/xkcd_machine_learning.png)
