"""
Tests for Semantic Search Assignment

These tests verify that:
1. Notes can be loaded from the text file
2. Embeddings have the correct shape
3. Similarity search returns ranked results
4. Results can be saved to JSON
"""

import pytest
import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class TestLoadNotes:
    """Test note loading functions."""

    def test_load_notes_exists(self):
        """Test that load_notes function exists."""
        from search import load_notes

        assert callable(load_notes)

    def test_load_notes_returns_list(self):
        """Test that load_notes returns a list."""
        from search import load_notes

        notes_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "clinical_notes.txt"
        )
        result = load_notes(notes_path)

        assert result is not None, "load_notes should return a list, not None"
        assert isinstance(result, list), "load_notes should return a list"

    def test_load_notes_correct_count(self):
        """Test that load_notes returns the right number of notes."""
        from search import load_notes

        notes_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "clinical_notes.txt"
        )
        result = load_notes(notes_path)

        assert result is not None
        assert len(result) == 4, f"Expected 4 notes, got {len(result)}"

    def test_load_notes_nonempty_strings(self):
        """Test that each note is a non-empty string."""
        from search import load_notes

        notes_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "clinical_notes.txt"
        )
        result = load_notes(notes_path)

        assert result is not None
        for i, note in enumerate(result):
            assert isinstance(note, str), f"Note {i} should be a string"
            assert len(note.strip()) > 0, f"Note {i} should not be empty"


class TestEmbedNotes:
    """Test embedding functions."""

    def test_embed_notes_exists(self):
        """Test that embed_notes function exists."""
        from search import embed_notes

        assert callable(embed_notes)

    def test_embed_notes_returns_array(self):
        """Test that embed_notes returns a numpy array with correct shape."""
        from search import embed_notes
        import numpy as np

        sample_notes = ["Patient has chest pain.", "Patient has diabetes."]
        result = embed_notes(sample_notes)

        assert result is not None, "embed_notes should return a numpy array"
        assert isinstance(result, np.ndarray), "embed_notes should return numpy array"
        assert result.shape[0] == 2, "Should have one embedding per note"
        assert result.shape[1] > 0, "Embedding dimension should be > 0"


class TestFindSimilar:
    """Test similarity search functions."""

    def test_find_similar_exists(self):
        """Test that find_similar function exists."""
        from search import find_similar

        assert callable(find_similar)

    def test_find_similar_returns_results(self):
        """Test that find_similar returns ranked results."""
        from search import embed_notes, find_similar

        notes = [
            "Patient with chest pain and elevated troponin.",
            "Patient with high blood sugar and diabetes.",
            "Patient with cough and lung infiltrate.",
        ]
        embeddings = embed_notes(notes)
        if embeddings is None:
            pytest.skip("embed_notes not implemented")

        results = find_similar("heart attack", notes, embeddings, top_k=2)

        assert results is not None, "find_similar should return results"
        assert isinstance(results, list), "find_similar should return a list"
        assert len(results) == 2, "Should return top_k results"

    def test_find_similar_result_structure(self):
        """Test that each result has 'note' and 'score' keys."""
        from search import embed_notes, find_similar

        notes = ["Chest pain patient.", "Diabetes patient."]
        embeddings = embed_notes(notes)
        if embeddings is None:
            pytest.skip("embed_notes not implemented")

        results = find_similar("cardiac", notes, embeddings, top_k=1)

        assert results is not None
        assert len(results) >= 1
        assert "note" in results[0], "Result should have 'note' key"
        assert "score" in results[0], "Result should have 'score' key"

    def test_find_similar_sorted_descending(self):
        """Test that results are sorted by score descending."""
        from search import embed_notes, find_similar

        notes = [
            "Patient with chest pain and elevated troponin.",
            "Patient with high blood sugar and diabetes.",
            "Patient with cough and lung infiltrate.",
        ]
        embeddings = embed_notes(notes)
        if embeddings is None:
            pytest.skip("embed_notes not implemented")

        results = find_similar("heart attack", notes, embeddings, top_k=3)

        assert results is not None
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), "Results should be sorted by score descending"


class TestSaveResults:
    """Test result saving functions."""

    def test_save_results_exists(self):
        """Test that save_results function exists."""
        from search import save_results

        assert callable(save_results)

    def test_save_results_creates_file(self):
        """Test that save_results creates a JSON file."""
        from search import save_results

        sample_results = [
            {"note": "Test note 1", "score": 0.95},
            {"note": "Test note 2", "score": 0.82},
        ]

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmppath = f.name

        try:
            # Remove the temp file so we can check if save_results creates it
            os.remove(tmppath)
            save_results(sample_results, tmppath)
            assert os.path.exists(tmppath), "save_results should create the output file"

            with open(tmppath) as f:
                loaded = json.load(f)
            assert isinstance(loaded, list), "Saved JSON should be a list"
            assert len(loaded) == 2, "Should save all results"
        finally:
            if os.path.exists(tmppath):
                os.remove(tmppath)


class TestDocumentation:
    """Test that documentation exists."""

    def test_readme_exists(self):
        """Test that README.md exists."""
        readme_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "README.md"
        )
        assert os.path.exists(readme_path), "README.md should exist"
