"""
Semantic Search Assignment: Clinical Note Search with Embeddings

Use sentence embeddings to search clinical notes by meaning rather than keywords.
"""

import json
import numpy as np
from typing import List, Dict


def load_notes(filepath: str = "clinical_notes.txt") -> List[str]:
    """
    Load clinical notes from a text file.

    The file has notes separated by '## Note N' headers.
    Parse out just the note text (skip headers and empty lines).

    Parameters
    ----------
    filepath : str
        Path to the clinical notes file

    Returns
    -------
    list of str
        List of clinical note strings
    """
    # TODO: Implement this function
    #
    # Hints:
    # - Read the file contents
    # - Split on "## Note" to separate individual notes
    # - Strip whitespace and skip empty strings
    # - The first split element will be the file header, skip it

    pass


def embed_notes(notes: List[str]) -> np.ndarray:
    """
    Generate embeddings for a list of clinical notes.

    Parameters
    ----------
    notes : list of str
        Clinical note strings to embed

    Returns
    -------
    numpy.ndarray
        Array of shape (n_notes, embedding_dim)
    """
    # TODO: Implement this function
    #
    # Hints:
    # - Use SentenceTransformer("all-MiniLM-L6-v2")
    # - model.encode(notes) returns a numpy array

    pass


def find_similar(
    query: str,
    notes: List[str],
    embeddings: np.ndarray,
    top_k: int = 2,
) -> List[Dict]:
    """
    Find the most similar notes to a query using cosine similarity.

    Parameters
    ----------
    query : str
        Search query
    notes : list of str
        Original note texts
    embeddings : numpy.ndarray
        Pre-computed embeddings for the notes
    top_k : int
        Number of top results to return

    Returns
    -------
    list of dict
        Top results as [{"note": str, "score": float}, ...] sorted by score descending
    """
    # TODO: Implement this function
    #
    # Hints:
    # - Embed the query with the same model
    # - Use sklearn.metrics.pairwise.cosine_similarity
    # - Sort by similarity score, return top_k results

    pass


def save_results(results: List[Dict], filepath: str = "search_results.json"):
    """
    Save search results to a JSON file.

    Parameters
    ----------
    results : list of dict
        Search results to save
    filepath : str
        Output file path
    """
    # TODO: Implement this function

    pass


if __name__ == "__main__":
    print("Loading clinical notes...")
    notes = load_notes("clinical_notes.txt")
    if notes:
        print(f"Loaded {len(notes)} notes")

        print("\nGenerating embeddings...")
        embeddings = embed_notes(notes)
        if embeddings is not None:
            print(f"Embeddings shape: {embeddings.shape}")

            query = "heart attack symptoms"
            print(f"\nSearching for: '{query}'")
            results = find_similar(query, notes, embeddings, top_k=2)

            if results:
                for i, r in enumerate(results, 1):
                    print(f"\n  {i}. (score: {r['score']:.3f})")
                    print(f"     {r['note'][:100]}...")

                save_results(results)
                print("\nResults saved to search_results.json")
