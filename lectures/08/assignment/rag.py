"""
RAG Assignment: Clinical Document Q&A

Complete the functions below to build a RAG pipeline for answering
questions based on clinical documents.
"""

import os
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import chromadb


# Initialize models (do this once)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()


def chunk_document(
    text: str, chunk_size: int = 500, overlap: int = 100
) -> List[str]:
    """
    Split a document into overlapping chunks.

    Parameters
    ----------
    text : str
        The document text to chunk
    chunk_size : int
        Target size of each chunk in characters
    overlap : int
        Number of overlapping characters between chunks

    Returns
    -------
    list of str
        List of text chunks
    """
    # TODO: Implement chunking
    #
    # Hints:
    # - Simple approach: split by character count with overlap
    # - Better approach: use LangChain's RecursiveCharacterTextSplitter
    #   which splits on natural boundaries (paragraphs, sentences)
    #
    # Example with LangChain:
    # from langchain.text_splitter import RecursiveCharacterTextSplitter
    # splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=chunk_size,
    #     chunk_overlap=overlap
    # )
    # return splitter.split_text(text)

    pass


def index_documents(
    documents: List[Dict], collection_name: str = "clinical_docs"
) -> None:
    """
    Index documents in the vector store.

    Parameters
    ----------
    documents : list of dict
        Each dict should have 'id', 'text', and optional 'metadata'
    collection_name : str
        Name of the ChromaDB collection
    """
    # TODO: Implement document indexing
    #
    # Steps:
    # 1. Create or get collection
    # 2. Chunk each document
    # 3. Generate embeddings for chunks
    # 4. Add to collection with metadata (source doc id, chunk index)
    #
    # Example:
    # collection = chroma_client.get_or_create_collection(name=collection_name)
    # for doc in documents:
    #     chunks = chunk_document(doc['text'])
    #     embeddings = embedding_model.encode(chunks).tolist()
    #     collection.add(
    #         documents=chunks,
    #         embeddings=embeddings,
    #         ids=[f"{doc['id']}_chunk_{i}" for i in range(len(chunks))],
    #         metadatas=[{"source": doc['id']} for _ in chunks]
    #     )

    pass


def retrieve(
    query: str, collection_name: str = "clinical_docs", n_results: int = 3
) -> List[Dict]:
    """
    Retrieve relevant chunks for a query.

    Parameters
    ----------
    query : str
        The search query
    collection_name : str
        Name of the ChromaDB collection
    n_results : int
        Number of results to return

    Returns
    -------
    list of dict
        Each dict contains 'text', 'metadata', and 'distance'
    """
    # TODO: Implement retrieval
    #
    # Steps:
    # 1. Get the collection
    # 2. Encode the query
    # 3. Query the collection
    # 4. Format and return results

    pass


def generate_answer(query: str, context: List[str]) -> str:
    """
    Generate an answer using the LLM with retrieved context.

    Parameters
    ----------
    query : str
        The user's question
    context : list of str
        Retrieved document chunks

    Returns
    -------
    str
        The generated answer
    """
    # TODO: Implement answer generation
    #
    # Steps:
    # 1. Format context into a prompt
    # 2. Call the LLM API
    # 3. Return the response
    #
    # Example prompt structure:
    # system: "Answer based only on the provided context..."
    # user: f"Context:\n{context}\n\nQuestion: {query}"

    pass


def rag_query(query: str, collection_name: str = "clinical_docs") -> Dict:
    """
    Complete RAG pipeline: retrieve relevant documents and generate an answer.

    Parameters
    ----------
    query : str
        The user's question
    collection_name : str
        Name of the ChromaDB collection

    Returns
    -------
    dict
        Contains 'answer', 'sources', and 'context'
    """
    # TODO: Implement the complete RAG pipeline
    #
    # Steps:
    # 1. Retrieve relevant chunks
    # 2. Extract context texts
    # 3. Generate answer
    # 4. Return results with sources

    pass


if __name__ == "__main__":
    # Example usage
    print("RAG Pipeline Test")
    print("-" * 50)

    # Sample documents for testing
    sample_docs = [
        {
            "id": "doc1",
            "text": "Hypertension is diagnosed when blood pressure consistently measures 130/80 mmHg or higher. First-line treatments include lifestyle modifications and medications such as ACE inhibitors or calcium channel blockers.",
        },
        {
            "id": "doc2",
            "text": "Type 2 diabetes is characterized by insulin resistance. Metformin is typically the first-line medication. Target HbA1c is generally less than 7% for most adults.",
        },
    ]

    # TODO: Test your implementation
    # 1. Index the sample documents
    # index_documents(sample_docs)

    # 2. Query the system
    # result = rag_query("What medication is used for diabetes?")
    # print(f"Answer: {result['answer']}")
