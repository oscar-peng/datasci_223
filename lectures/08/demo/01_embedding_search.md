# Demo 1: Embedding Similarity Search

In this demo, we'll build a semantic search system using embeddings to find similar clinical documents.

## Learning Objectives

- Generate embeddings from text using Sentence Transformers
- Calculate similarity between documents
- Build a simple semantic search function
- Understand when embeddings outperform keyword search

## Setup

```python
# %% Setup
# Install if needed: pip install sentence-transformers scikit-learn numpy

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
```

## Sample Clinical Notes

```python
# %% Sample data
clinical_notes = [
    "Patient presents with acute chest pain radiating to left arm. ECG shows ST elevation. Troponin elevated at 2.1 ng/mL. Diagnosis: STEMI.",
    "65-year-old female with uncontrolled type 2 diabetes. HbA1c 9.4%. Current medications: metformin 1000mg BID, lisinopril 10mg daily.",
    "Follow-up visit for hypertension management. Blood pressure 142/88. Increased amlodipine from 5mg to 10mg daily.",
    "Patient reports persistent cough for 2 weeks. Chest X-ray shows right lower lobe infiltrate. Started on azithromycin for community-acquired pneumonia.",
    "Annual physical exam. Labs within normal limits. BMI 24.2. Colonoscopy due next year.",
    "Emergency room visit for severe headache, photophobia, and neck stiffness. Lumbar puncture pending. Concern for meningitis.",
    "Post-operative day 2 following laparoscopic cholecystectomy. Tolerating clear liquids. Pain controlled with oral analgesics.",
    "Cardiology consult for preoperative evaluation. Patient has history of atrial fibrillation on warfarin. EKG shows rate-controlled afib.",
]
```

## Generate Embeddings

```python
# %% Load model and generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for all documents
note_embeddings = model.encode(clinical_notes)

print(f"Number of documents: {len(clinical_notes)}")
print(f"Embedding dimension: {note_embeddings.shape[1]}")
print(f"Embedding matrix shape: {note_embeddings.shape}")
```

## Semantic Search Function

```python
# %% Build search function
def semantic_search(query, documents, embeddings, model, top_k=3):
    """
    Find the most similar documents to a query using cosine similarity.
    
    Parameters
    ----------
    query : str
        Search query
    documents : list
        List of document strings
    embeddings : np.array
        Pre-computed document embeddings
    model : SentenceTransformer
        Model to encode the query
    top_k : int
        Number of results to return
    
    Returns
    -------
    list of tuples
        (document, similarity_score) for top matches
    """
    # Encode the query
    query_embedding = model.encode([query])
    
    # Calculate similarities
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Return results with scores
    results = []
    for idx in top_indices:
        results.append((documents[idx], similarities[idx]))
    
    return results
```

## Test Semantic Search

```python
# %% Test with various queries

# Query 1: Cardiac-related
print("=" * 60)
print("Query: 'heart attack symptoms'")
print("=" * 60)
results = semantic_search("heart attack symptoms", clinical_notes, note_embeddings, model)
for i, (doc, score) in enumerate(results, 1):
    print(f"\n{i}. (Score: {score:.3f})")
    print(f"   {doc[:100]}...")

# Query 2: Diabetes management
print("\n" + "=" * 60)
print("Query: 'blood sugar control'")
print("=" * 60)
results = semantic_search("blood sugar control", clinical_notes, note_embeddings, model)
for i, (doc, score) in enumerate(results, 1):
    print(f"\n{i}. (Score: {score:.3f})")
    print(f"   {doc[:100]}...")

# Query 3: Respiratory infection
print("\n" + "=" * 60)
print("Query: 'lung infection treatment'")
print("=" * 60)
results = semantic_search("lung infection treatment", clinical_notes, note_embeddings, model)
for i, (doc, score) in enumerate(results, 1):
    print(f"\n{i}. (Score: {score:.3f})")
    print(f"   {doc[:100]}...")
```

## Compare to Keyword Search

```python
# %% Compare semantic vs keyword search

def keyword_search(query, documents, top_k=3):
    """Simple keyword matching search."""
    query_words = set(query.lower().split())
    scores = []
    
    for doc in documents:
        doc_words = set(doc.lower().split())
        # Count matching words
        matches = len(query_words.intersection(doc_words))
        scores.append(matches)
    
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(documents[idx], scores[idx]) for idx in top_indices]

# Test query where semantic search excels
query = "cardiac emergency"

print("Semantic search results for 'cardiac emergency':")
for doc, score in semantic_search(query, clinical_notes, note_embeddings, model):
    print(f"  Score: {score:.3f} - {doc[:60]}...")

print("\nKeyword search results for 'cardiac emergency':")
for doc, score in keyword_search(query, clinical_notes):
    print(f"  Matches: {score} - {doc[:60]}...")
```

## Exercises

1. **Add more documents**: Add your own clinical notes and test how well the search performs
2. **Try different queries**: Test queries that use synonyms or related concepts
3. **Experiment with models**: Try `all-mpnet-base-v2` (more accurate but slower) or `paraphrase-MiniLM-L6-v2`
4. **Calculate document similarity**: Build a matrix of all pairwise document similarities

## Key Takeaways

- Embeddings capture semantic meaning, not just keywords
- Cosine similarity is a standard metric for comparing embeddings
- Semantic search finds relevant documents even without exact keyword matches
- Pre-computing embeddings makes search fast
