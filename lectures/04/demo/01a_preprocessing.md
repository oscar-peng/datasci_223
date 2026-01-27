---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Demo 1a: Text Preprocessing Safari

**Goal:** See how preprocessing choices affect downstream analysis; build a reusable `preprocess(text)` function.  
**Text:** Alice in Wonderland, Chapter 1 (~2,000 words).  
**Lecture:** Tokenization, Normalization, Stemming and Lemmatization.

---

## Setup: all imports and NLTK/spaCy

Run this cell first. All libraries and the spaCy model are loaded here so later cells focus on the analysis. Paths and the spaCy model name come from `config.yaml` so you can point at another data dir or model without editing code.

```python
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import spacy
import yaml
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

DEMO_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
CONFIG_PATH = DEMO_DIR / "config.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

DATA_DIR = DEMO_DIR / CONFIG["data"]["dir"]
nlp = spacy.load(CONFIG["spacy"]["model"])
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
```

---

## Load Alice excerpt

We load the raw text and inspect its size and the first few lines—this is the *input* our pipeline will transform. The file path comes from `config.yaml` (`data.files.alice`).

```python
alice_path = DATA_DIR / CONFIG["data"]["files"]["alice"]
raw_text = alice_path.read_text(encoding="utf-8")

print(f"Length: {len(raw_text)} chars, ~{len(raw_text.split())} words")
print(raw_text[:500])
```

---

## Tokenization: compare methods

**Why it matters:** Different tokenizers treat contractions, punctuation, and hyphenated words differently. Downstream steps (counts, search) depend on these choices.

We use one sentence with edge cases—*didn't*, *rabbit-hole*, punctuation—and compare three methods so you can see exactly how each splits the string.

```python
sentence = "Alice didn't think it so very much out of the way to hear the Rabbit say to itself, 'Oh dear! Oh dear! I shall be late!'"
print("Sample:", sentence)
print()

# Naive split: keeps "didn't" as one token, punctuation attached
print("text.split():", sentence.split())
print()

# NLTK: splits contractions and punctuation
tokens_nltk = nltk.word_tokenize(sentence)
print("nltk.word_tokenize:", tokens_nltk)
print()

# spaCy: one tokenizer per pipeline; tokens have .text
doc = nlp(sentence)
tokens_spacy = [t.text for t in doc]
print("spaCy tokens:", tokens_spacy)
```

---

## Sentence tokenization (optional use case)

When you need sentence boundaries (e.g. for sentence-level sentiment or extraction), use `nltk.sent_tokenize`. Word tokenizers don't split on sentence boundaries.

```python
sentences = nltk.sent_tokenize(raw_text[:800])
print(f"First 3 sentences: {len(sentences)} extracted")
for i, s in enumerate(sentences[:3]):
    print(f"  {i+1}. {s[:60]}...")
```

---

## Normalization: lowercase and stopwords

**Why it matters:** Lowercasing collapses "Alice" and "alice" into one type; that helps frequency counts but loses the cue that Alice is a name. Stopword removal shrinks the vocabulary but can remove important negation ("not", "no")—e.g. "No chest pain" in clinical notes. A **common mistake** in clinical NLP is dropping "no"/"not" and reversing meaning.

For some tasks you also remove punctuation (e.g. `string.punctuation`), but keep it when it carries meaning (e.g. hyphens in "COVID-19").

Here we apply lowercase and show how many tokens remain with and without the default English stopword list.

```python
lowered = raw_text.lower()
tokens = nltk.word_tokenize(lowered)
print("Token count (lowercased):", len(tokens))

stop_words = set(stopwords.words("english"))
print("'not' in stopwords?", "not" in stop_words)
print("'no' in stopwords?", "no" in stop_words)

tokens_no_stop = [t for t in tokens if t not in stop_words]
print("After removing stopwords:", len(tokens_no_stop))
```

---

## Stem vs Lemma

**Why it matters:** Stemming uses rules and can produce non-words ("studies" → "studi"); lemmatization uses a dictionary and needs part-of-speech for words like "better" → "good". Both reduce surface forms to a single base so we can count variants together.

We run stemmer and lemmatizer on a few example words so you can see the tradeoff.

```python
words = ["curiouser", "running", "studies", "better"]
for w in words:
    stem = stemmer.stem(w)
    lemma_n = lemmatizer.lemmatize(w, pos="n")
    lemma_a = lemmatizer.lemmatize(w, pos="a")
    print(f"  {w}: stem={stem}, lemma(n)={lemma_n}, lemma(a)={lemma_a}")
```

---

## Build a preprocess function

We combine tokenization, optional lowercase/stopword removal, and lemmatization into one reusable function. Later we'll use its output for a visual summary of the vocabulary.

```python
def preprocess(
    text: str,
    *,
    lowercase: bool = True,
    remove_stop: bool = True,
    lemmatize: bool = True,
    stop_words: set | None = None,
) -> list[str]:
    if stop_words is None and remove_stop:
        stop_words = set(stopwords.words("english"))
    if lowercase:
        text = text.lower()
    tokens = nltk.word_tokenize(text)
    if remove_stop and stop_words:
        tokens = [t for t in tokens if t not in stop_words]
    if not lemmatize:
        return tokens
    out = []
    for t in tokens:
        lemma = lemmatizer.lemmatize(t, pos="v")
        if lemma == t:
            lemma = lemmatizer.lemmatize(t, pos="n")
        if lemma == t:
            lemma = lemmatizer.lemmatize(t, pos="a")
        out.append(lemma)
    return out

first_para = raw_text.split("\n\n")[0]
print("First paragraph tokens (preprocessed):", preprocess(first_para)[:25])
```

---

## Visual: top tokens after preprocessing

Seeing the most frequent tokens as a bar chart makes the effect of preprocessing concrete: common words dominate, and the shape of the distribution reflects our choices (e.g. no stopwords, lemmatized). The number of top terms is set in `config.yaml` (`visualization.top_tokens`).

```python
tokens = preprocess(raw_text)
counts = Counter(tokens)
top_n = CONFIG["visualization"]["top_tokens"]
top = counts.most_common(top_n)
terms, freqs = zip(*top)

fig, ax = plt.subplots(figsize=(10, 4))
ax.barh(range(len(terms)), freqs, align="center")
ax.set_yticks(range(len(terms)))
ax.set_yticklabels(terms)
ax.invert_yaxis()
ax.set_xlabel("Count")
ax.set_title(f"Top {top_n} tokens in Alice Ch. 1 (preprocessed: lowercase, no stopwords, lemmatized)")
plt.tight_layout()
plt.show()
```
