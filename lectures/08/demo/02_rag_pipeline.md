# Demo 2: Simple RAG Pipeline

In this demo, we'll build a Retrieval-Augmented Generation (RAG) pipeline for clinical document Q&A.

## Learning Objectives

- Understand the RAG pipeline components
- Build a simple vector store
- Combine retrieval with generation
- Evaluate RAG response quality

## Setup

```python
# %% Setup
# Install if needed: 
# pip install sentence-transformers chromadb openai python-dotenv

import os
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI

# Load API key from environment
# Set OPENAI_API_KEY in your environment or .env file
client = OpenAI()
```

## Clinical Knowledge Base

```python
# %% Create sample knowledge base

# Simulating chunks from clinical guidelines
clinical_knowledge = [
    {
        "id": "hypertension_1",
        "text": "Stage 1 hypertension is defined as systolic blood pressure 130-139 mmHg or diastolic 80-89 mmHg. Initial treatment includes lifestyle modifications: weight loss, DASH diet, sodium restriction (<2300mg/day), regular exercise (150 min/week moderate intensity).",
        "source": "JNC8 Guidelines"
    },
    {
        "id": "hypertension_2", 
        "text": "First-line antihypertensive medications include thiazide diuretics, ACE inhibitors, ARBs, and calcium channel blockers. Choice depends on comorbidities: ACE inhibitors or ARBs preferred in diabetes and chronic kidney disease.",
        "source": "JNC8 Guidelines"
    },
    {
        "id": "diabetes_1",
        "text": "Type 2 diabetes is diagnosed with fasting glucose ≥126 mg/dL, HbA1c ≥6.5%, or 2-hour glucose ≥200 mg/dL during OGTT. Metformin is first-line therapy unless contraindicated. Target HbA1c <7% for most adults.",
        "source": "ADA Standards of Care"
    },
    {
        "id": "diabetes_2",
        "text": "For patients with type 2 diabetes and cardiovascular disease, SGLT2 inhibitors or GLP-1 receptor agonists with proven cardiovascular benefit are recommended regardless of HbA1c. Examples: empagliflozin, liraglutide.",
        "source": "ADA Standards of Care"
    },
    {
        "id": "chest_pain_1",
        "text": "Acute chest pain evaluation: HEART score assesses History, ECG, Age, Risk factors, and Troponin. Score 0-3 is low risk (discharge with follow-up), 4-6 is intermediate (admit for observation), 7-10 is high risk (early invasive strategy).",
        "source": "AHA Chest Pain Guidelines"
    },
    {
        "id": "chest_pain_2",
        "text": "STEMI diagnosis requires ST elevation ≥1mm in 2 contiguous leads or new LBBB with symptoms. Door-to-balloon time goal <90 minutes for primary PCI. If PCI not available within 120 minutes, fibrinolysis within 30 minutes.",
        "source": "AHA Chest Pain Guidelines"
    },
]
```

## Initialize Vector Store

```python
# %% Set up ChromaDB and embeddings

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB (in-memory for demo)
chroma_client = chromadb.Client()

# Create collection
collection = chroma_client.create_collection(
    name="clinical_guidelines",
    metadata={"description": "Clinical practice guidelines"}
)

# Add documents to collection
documents = [doc["text"] for doc in clinical_knowledge]
ids = [doc["id"] for doc in clinical_knowledge]
metadatas = [{"source": doc["source"]} for doc in clinical_knowledge]
embeddings = embedding_model.encode(documents).tolist()

collection.add(
    documents=documents,
    ids=ids,
    metadatas=metadatas,
    embeddings=embeddings
)

print(f"Added {collection.count()} documents to the knowledge base")
```

## RAG Query Function

```python
# %% RAG query function

def rag_query(question, n_results=3, show_sources=True):
    """
    Answer a question using RAG.
    
    Parameters
    ----------
    question : str
        The question to answer
    n_results : int
        Number of documents to retrieve
    show_sources : bool
        Whether to print retrieved sources
    
    Returns
    -------
    str
        The generated answer
    """
    # Step 1: Retrieve relevant documents
    query_embedding = embedding_model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    
    if show_sources:
        print("Retrieved documents:")
        for i, (doc, meta, dist) in enumerate(zip(
            results['documents'][0], 
            results['metadatas'][0],
            results['distances'][0]
        )):
            print(f"  {i+1}. [{meta['source']}] (distance: {dist:.3f})")
            print(f"     {doc[:100]}...")
        print()
    
    # Step 2: Build context from retrieved documents
    context = "\n\n".join(results['documents'][0])
    
    # Step 3: Generate answer with LLM
    system_prompt = """You are a helpful clinical assistant. Answer the question based ONLY on the provided context. 
If the context doesn't contain enough information to answer, say so.
Always mention which guideline or source the information comes from."""

    user_prompt = f"""Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    
    return response.choices[0].message.content
```

## Test RAG Pipeline

```python
# %% Test with clinical questions

# Question 1: Hypertension treatment
print("=" * 60)
question1 = "What is the first-line treatment for a patient with newly diagnosed hypertension and diabetes?"
print(f"Question: {question1}\n")
answer1 = rag_query(question1)
print(f"Answer: {answer1}")

# Question 2: Chest pain evaluation
print("\n" + "=" * 60)
question2 = "How do I evaluate a patient with acute chest pain?"
print(f"Question: {question2}\n")
answer2 = rag_query(question2)
print(f"Answer: {answer2}")

# Question 3: Diabetes diagnosis
print("\n" + "=" * 60)
question3 = "What HbA1c level indicates diabetes?"
print(f"Question: {question3}\n")
answer3 = rag_query(question3)
print(f"Answer: {answer3}")
```

## Compare RAG vs Direct LLM

```python
# %% Compare RAG to direct LLM query

def direct_llm_query(question):
    """Ask LLM directly without retrieval."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a clinical assistant. Be concise."},
            {"role": "user", "content": question}
        ],
        temperature=0
    )
    return response.choices[0].message.content

# Test comparison
test_question = "What blood pressure defines stage 1 hypertension according to current guidelines?"

print("RAG Response:")
print(rag_query(test_question, show_sources=True))

print("\nDirect LLM Response:")
print(direct_llm_query(test_question))
```

## RAG with Source Citations

```python
# %% Enhanced RAG with citations

def rag_with_citations(question, n_results=3):
    """RAG that includes source citations in the response."""
    
    query_embedding = embedding_model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=["documents", "metadatas"]
    )
    
    # Build numbered context
    context_parts = []
    sources = []
    for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        context_parts.append(f"[{i+1}] {doc}")
        sources.append(f"[{i+1}] {meta['source']}")
    
    context = "\n\n".join(context_parts)
    
    system_prompt = """You are a clinical assistant. Answer based on the provided numbered sources.
Include citation numbers [1], [2], etc. when referencing information from specific sources."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0
    )
    
    answer = response.choices[0].message.content
    
    return {
        "answer": answer,
        "sources": sources
    }

# Test
print("=" * 60)
result = rag_with_citations("What medications are recommended for diabetic patients with heart disease?")
print(f"Answer: {result['answer']}")
print(f"\nSources used:")
for source in result['sources']:
    print(f"  {source}")
```

## Exercises

1. **Add more documents**: Expand the knowledge base with additional guidelines
2. **Chunking strategy**: Split longer documents into smaller chunks
3. **Hybrid search**: Combine semantic search with keyword filtering
4. **Evaluation**: Create test questions with known answers to evaluate accuracy

## Key Takeaways

- RAG grounds LLM responses in actual documents
- Vector databases enable fast semantic retrieval
- Citation tracking increases trustworthiness
- Context quality directly affects answer quality
