# Lecture 04 Notes — Natural Language Processing

## What is NLP?

The fundamental tension in NLP is that human language evolved for human brains, not computers. We communicate with ambiguity *on purpose* — it's a feature, not a bug. When you say "I saw her duck," a human uses context (is she at a pond? at a bar?) to resolve the meaning instantly. A computer sees a subject, a verb, and then... is "duck" a noun or a verb? Is "her" possessive or objective?

Language is also deeply *redundant*. In English, roughly 50% of characters in a sentence are predictable from context alone (Shannon's entropy experiments from the 1950s showed this). NLP exploits that redundancy — statistical patterns in how words co-occur let algorithms make surprisingly good guesses about meaning without truly "understanding" anything.

The clinical context makes this harder. A doctor writing "pt c/o SOB" is using a compression scheme (abbreviations) that saves time but loses the redundancy NLP algorithms rely on. Clinical text is the hardest NLP domain because it combines extreme abbreviation, domain-specific jargon, critical negation ("denies CP"), and life-or-death stakes for getting it wrong.

## Why NLP for Health Data?

About 80% of clinical data is unstructured text — progress notes, radiology reports, discharge summaries. ICD codes and structured fields capture the billing story, but the clinical story lives in free text. A patient's chart might have an ICD code for "chest pain" but the nuance — was it exertional? substernal? relieved by rest? — is in the note.

Practical example: identifying patients who experienced an adverse drug reaction. Structured data might show the drug was prescribed and later discontinued, but the *reason* for discontinuation (rash? GI upset? patient preference?) is buried in a note. NLP lets you systematically extract that reason across thousands of charts.

Another concrete use: cohort building for clinical trials. "Find all patients with treatment-resistant depression" requires reading notes to determine that multiple medications were tried and failed — something that structured diagnosis codes alone can't reliably capture.

## Classical vs LLM-based Approaches

Classical NLP is *glass box* engineering — you can open it up, see every gear, and explain exactly why the system made a decision. When a classical pipeline says "this note mentions chest pain," you can trace it: the tokenizer split the text, the TF-IDF vectorizer weighted "chest" and "pain," and the classifier threshold was 0.7. That interpretability matters for clinical applications where you need to explain decisions to clinicians, IRBs, and regulators.

LLMs work more like a black box that happens to be extremely good at pattern recognition. GPT-style models learn contextual representations where the same word gets different vectors depending on surrounding words — "bank" near "river" vs "bank" near "account" get different internal representations. That contextual understanding is powerful but opaque.

The practical tradeoff: classical NLP is cheap, fast, interpretable, and works with small labeled datasets. LLMs are expensive, slow, opaque, but handle nuance and generalization much better. Most production clinical NLP systems today use classical techniques for structured extraction (pulling out vitals, medications, dates) and reserve LLMs for tasks requiring deeper understanding (summarization, question answering, complex reasoning about patient trajectories).

## Tools: NLTK vs spaCy

NLTK was built by academics for academics. Steven Bird and Edward Loper created it in 2001 at the University of Pennsylvania as a teaching tool. It exposes the internals — you can swap in different tokenizers, stemmers, taggers, and see how each algorithm behaves differently on the same input. That transparency is why we use it in class.

spaCy was built by Matthew Honnibal (a computational linguist) starting in 2015 specifically for production use. Instead of offering five tokenizers and letting you choose, spaCy gives you one that's been optimized for speed and accuracy. The object model (`Doc` → `Token` → `Span`) means you process text once and every annotation (POS, NER, dependency parse, lemma) is immediately available on the same objects.

The processing model difference matters: NLTK works on strings and lists, so you're constantly converting between representations. spaCy processes everything into a `Doc` object in one pass — the pipeline runs tokenization, tagging, parsing, and NER sequentially, and each component annotates the same data structure.

## Tokenization

Tokenization seems trivial until you try it. The sentence "Dr. Smith prescribed 500mg ibuprofen" has at least three reasonable tokenizations depending on your downstream task:

- If you're counting word frequencies, you probably want "500mg" as one token
- If you're extracting dosages, you need "500" and "mg" separated
- If you're doing NER, you need "Dr." to stay together (it's a title, not a sentence-ending period)

Subword tokenization (used by modern LLMs) takes a different approach entirely. Instead of splitting on whitespace and punctuation, BPE (Byte Pair Encoding) builds a vocabulary from character pairs that frequently co-occur. So "tokenization" might become "token" + "ization" — the model learns morphological structure implicitly. This elegantly handles rare words and typos: even if "cardiomyopathy" isn't in the vocabulary, its subpieces probably are.

The critical point: your tokenizer and your model must match. You can't tokenize with NLTK's word tokenizer and feed the result to a BERT model that expects WordPiece tokens. The token boundaries would be wrong and every downstream prediction would be garbage.

## Normalization

Lowercasing is a lossy compression. You're trading information (case) for simplicity (smaller vocabulary). In general English, that's usually fine — "The" and "the" are the same word. But clinical text is case-sensitive in ways that matter: "MS" (multiple sclerosis) vs "ms" (milliseconds), "US" (ultrasound) vs "us" (pronoun), "ED" (emergency department) vs "ed" (past tense suffix).

Stopword removal is similarly context-dependent. NLTK's English stopword list has 179 words. For topic modeling or document similarity, removing them focuses the signal on content words. But for clinical NLP, negation words ("no," "not," "without," "denies") are stopwords in the NLTK list that carry critical clinical meaning. "No chest pain" with stopwords removed becomes "chest pain" — the opposite meaning.

The general principle: preprocessing is not neutral. Every transformation you apply is a modeling decision that assumes certain information is irrelevant. The right preprocessing depends entirely on what you're trying to do downstream.

## Stemming and Lemmatization

Stemming is a heuristic: chop off suffixes according to rules. Porter's stemmer (1980) uses about 60 rules like "if the word ends in -ies and has more than one syllable, replace with -i." Fast, simple, language-specific. The downside: "university" → "univers," "studies" → "studi" — these aren't real words, so you can't look them up in a dictionary or display them to a human.

Lemmatization is a lookup: find the dictionary headword. "better" → "good" (if you know it's an adjective), "ran" → "run" (if you know it's a verb). This requires knowing the part of speech — "meeting" could lemmatize to "meeting" (noun) or "meet" (verb). spaCy handles this automatically because its pipeline runs POS tagging before lemmatization.

When to use which: stemming is fine when you just need to group related words and don't care about readability (e.g., building features for a classifier). Lemmatization is better when you need to display results to humans or when exact word forms matter (e.g., matching against a medical terminology database like UMLS).

## Part-of-Speech Tagging

POS tagging is one of those NLP tasks that seems academic until you need it. Two practical uses:

1. **Disambiguation for lemmatization**: "running" as a verb lemmatizes to "run"; as a noun ("running of the bulls") it stays "running." The POS tag tells the lemmatizer which form to produce.

2. **Information extraction**: If you want to find the main topics in a clinical note, extracting nouns gives you a useful first pass. Adjective + noun patterns ("severe pain," "chronic fatigue," "acute onset") capture clinically meaningful descriptors.

Penn Treebank tags (used by NLTK) are fine-grained: NN (singular noun), NNS (plural noun), NNP (proper noun singular), NNPS (proper noun plural). Universal Dependencies tags (used by spaCy) are coarser: all of those map to just "NOUN" or "PROPN." The fine-grained tags give you more information; the universal tags are easier to work with and consistent across languages.

Modern POS taggers achieve ~97% accuracy on well-edited English text. On clinical text, accuracy drops to ~93-95% because of sentence fragments, abbreviations, and non-standard grammar ("pt a&o x3, no acute distress").

## Named Entity Recognition

NER is pattern recognition for proper nouns and structured concepts. Standard NER models recognize a fixed set of entity types — PERSON, ORG, GPE (geopolitical entity), DATE, MONEY, etc. These categories were defined by the Message Understanding Conferences (MUC) in the 1990s for news text.

Clinical text needs different entity types: MEDICATION, DOSAGE, DIAGNOSIS, PROCEDURE, ANATOMY. Standard NER models won't find these because they weren't trained on clinical text. That's why specialized models like scispaCy and MedSpaCy exist — they're trained on annotated biomedical and clinical corpora.

A key limitation: NER works on mentions, not resolution. If a note says "Dr. Smith" and later "the attending physician," NER identifies both as entities but doesn't know they're the same person. That's a separate task called coreference resolution, which is much harder.

NLTK's NER uses a tree structure because named entities can span multiple tokens ("New York City" is one GPE, not three separate words). Each entity is a subtree where the leaves are the tokens and the label is the entity type. spaCy represents this more cleanly with `Span` objects that have `.text` and `.label_` attributes.

## Regex Patterns

Regular expressions are the Swiss army knife of text extraction. NER finds semantic entities (this is a person, this is a date). Regex finds structural patterns (three digits, a slash, two digits).

For clinical text, regex is indispensable for:
- **Vitals**: `\d{2,3}/\d{2,3}` matches blood pressure readings like "120/80"
- **Dosages**: `\d+\s?(mg|ml|mcg|units)` matches "500mg", "10 ml", "100 units"
- **Lab values**: `\d+\.?\d*\s?(mg/dL|mmol/L|mEq/L)` matches lab results with units
- **Dates in various formats**: clinical systems use every date format imaginable

The power of regex is composability — you can build complex patterns from simple pieces. The danger is fragility — a regex that matches "500mg" won't match "500 mg" (with a space) unless you explicitly account for optional whitespace.

Regex101.com is a genuinely useful tool for building and testing patterns interactively. The explanation panel shows exactly what each part of the pattern does, which is invaluable when debugging.

## Bag of Words and TF-IDF

Bag of Words is the simplest text-to-numbers conversion: count how many times each word appears in each document. The result is a document-term matrix — rows are documents, columns are words, values are counts.

The fundamental assumption: word order doesn't matter. "Patient denies pain" and "Pain denies patient" produce identical vectors. This is obviously wrong linguistically, but surprisingly useful in practice for tasks like document classification and clustering. The statistical distribution of words is often enough to distinguish document types.

TF-IDF adds one important insight: words that appear in every document aren't useful for distinguishing between documents. "Patient" appears in every clinical note — high term frequency, low discriminative value. "Metformin" appears in only diabetes-related notes — that's informative. IDF (inverse document frequency) mathematically downweights ubiquitous words and upweights distinctive ones.

The formula: TF-IDF = TF × IDF, where IDF = log(total documents / documents containing the word). A word in every document gets IDF = log(1) = 0, zeroing out its contribution entirely. A word in only one document gets the maximum IDF weight.

scikit-learn's `TfidfVectorizer` uses a slightly different formula with smoothing (adds 1 to numerator and denominator in IDF) to avoid division by zero and to prevent rare words from dominating too aggressively.

## N-grams

Single words (unigrams) lose context that matters. "Chest pain" as a bigram is a clinical concept; "chest" and "pain" separately are just body parts and symptoms. "No chest pain" as a trigram captures negation; the unigram "pain" doesn't.

N-grams are a simple way to capture local word order without building a full syntax model. The tradeoff is vocabulary explosion: if you have V unique words, you can have up to V² unique bigrams and V³ unique trigrams. In practice, most possible n-grams never occur, so the matrix is extremely sparse.

The `ngram_range=(1, 2)` parameter in scikit-learn's vectorizers means "include both unigrams and bigrams." This is a common practical choice — you get individual word signals plus the most important two-word phrases.

For clinical text, bigrams and trigrams are particularly valuable because so much clinical language is multi-word: "chest pain," "shortness of breath," "blood pressure," "heart rate," "white blood cell count."

## Word Vectors

Bag of Words and TF-IDF treat every word as equally different from every other word. "Diabetes" and "hypertension" are just as distant as "diabetes" and "pizza." Word vectors (embeddings) fix this by mapping words into a continuous space where semantic similarity corresponds to geometric proximity.

The intuition: words that appear in similar contexts should have similar meanings. "Diabetes" and "hypertension" both appear near words like "patient," "management," "chronic," "medication." "Pizza" appears near "delivery," "toppings," "cheese." The embedding algorithm learns these co-occurrence patterns and encodes them as dense vectors (typically 100-300 dimensions).

spaCy's small model (`en_core_web_sm`) doesn't include word vectors — it's optimized for pipeline speed. The medium (`_md`) and large (`_lg`) models include 300-dimensional GloVe vectors. For this lecture, word vectors are a conceptual bridge to Lecture 07 where we'll cover embeddings in depth.

## Document Similarity

Cosine similarity measures the angle between two vectors, not the distance. This is important because document length affects vector magnitude but not direction. A 10-page discharge summary and a 1-paragraph progress note about the same condition will have very different vector magnitudes (because the longer document has higher word counts) but similar directions (because the relative proportions of words are similar).

Cosine similarity ranges from 0 to 1 for TF-IDF vectors (because all values are non-negative). A score of 1.0 means the documents use exactly the same words in exactly the same proportions. A score of 0.0 means they share no words at all. In practice, even very similar documents rarely exceed 0.5-0.7 because natural language variation introduces many unique word choices.

Practical applications in health data: finding similar patient notes (for case-based reasoning), deduplicating records, building search systems over clinical text, clustering discharge summaries by condition type.

## Specialized Tools (scispaCy, MedSpaCy, UMLS)

General NLP models are trained on news, web text, and Wikipedia. Clinical text is different enough that these models underperform. scispaCy provides models trained on biomedical literature (PubMed abstracts, PMC full texts) — they recognize entities like genes, chemicals, diseases, and cell types that general models miss entirely.

MedSpaCy adds clinical-specific pipeline components, most importantly negation detection. The NegEx algorithm (Chapman et al., 2001) is a simple but effective rule-based approach: it looks for negation triggers ("no," "denies," "without," "ruled out") and checks whether they're within a window of a clinical finding. "No chest pain" → chest pain is negated. "History of MI, no recurrence" → MI is affirmed, recurrence is negated.

UMLS is the Rosetta Stone of medical terminologies. It maps between ICD codes, SNOMED CT, RxNorm, MeSH, and dozens of other coding systems. When NLP extracts "heart attack" from text, UMLS can map that to the SNOMED concept for "myocardial infarction" (22298006), which links to the ICD-10 code I21. This concept normalization is essential for combining NLP-extracted information with structured clinical data.

## Challenges in Clinical NLP

The four horsemen of clinical NLP:

**Negation** is the most critical. A study by Chapman et al. found that up to 50% of clinical findings mentioned in radiology reports are negated. Simple keyword extraction without negation detection will have an unacceptably high false positive rate. "No evidence of malignancy" is a *good* finding, not a cancer diagnosis.

**Abbreviation ambiguity** is pervasive. A study of clinical abbreviations found that the average abbreviation has 3.5 possible meanings. "MS" alone has at least 12 medical meanings. Context usually disambiguates for a human reader, but automatic disambiguation requires models trained on clinical text.

**Uncertainty** is clinically important but linguistically subtle. "Possible pneumonia" vs "pneumonia" vs "no pneumonia" represent three different clinical states. Hedge words ("possible," "likely," "suspected," "cannot rule out") are common in clinical notes because physicians often document differential diagnoses, not definitive ones.

**Temporality** distinguishes current conditions from history. "History of breast cancer" (past, possibly resolved) vs "breast cancer" (current) vs "family history of breast cancer" (not the patient's condition at all) require temporal and relational reasoning that basic NLP pipelines don't handle.

## Pipelines: NLTK vs spaCy

The NLTK pipeline is like cooking from scratch — you choose every ingredient and control every step. Tokenize, then filter stopwords, then lemmatize, then POS tag, then NER. If you want to change one step (different stopword list, different stemmer), you just swap that component. The cost is boilerplate code and slower execution.

The spaCy pipeline is like a food processor — one button does everything. `doc = nlp(text)` runs tokenization, POS tagging, dependency parsing, lemmatization, and NER in a single optimized pass. Results are all accessible as attributes on the same `Doc`/`Token` objects. The cost is less granular control (though you can disable or customize components).

The choice depends on the task: use NLTK when you're exploring and need to understand what each step does (which is why we start with it in class). Use spaCy when you need to process text at scale and want reliable, fast, production-quality results.

Both pipelines leave regex as a manual addition — neither NLTK nor spaCy automatically extracts vitals, dosages, or other format-based patterns. That's where `re.findall()` complements the NLP pipeline.
