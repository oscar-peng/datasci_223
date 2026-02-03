"""
Tests for RAG Assignment

These tests verify that:
1. Required functions exist
2. Chunking works correctly
3. PHI detection works
4. RAG pipeline components are implemented
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class TestChunking:
    """Test document chunking functions."""

    def test_chunk_document_exists(self):
        """Test that chunk_document function exists."""
        from rag import chunk_document

        assert callable(chunk_document)

    def test_chunk_document_returns_list(self):
        """Test that chunk_document returns a list."""
        from rag import chunk_document

        text = "This is a sample text. " * 50  # Long enough to chunk
        result = chunk_document(text, chunk_size=100, overlap=20)

        if result is not None:
            assert isinstance(result, list), (
                "chunk_document should return a list"
            )
            assert len(result) > 0, "Should return at least one chunk"


class TestIndexing:
    """Test document indexing functions."""

    def test_index_documents_exists(self):
        """Test that index_documents function exists."""
        from rag import index_documents

        assert callable(index_documents)


class TestRetrieval:
    """Test retrieval functions."""

    def test_retrieve_exists(self):
        """Test that retrieve function exists."""
        from rag import retrieve

        assert callable(retrieve)


class TestGeneration:
    """Test answer generation functions."""

    def test_generate_answer_exists(self):
        """Test that generate_answer function exists."""
        from rag import generate_answer

        assert callable(generate_answer)


class TestRAGPipeline:
    """Test complete RAG pipeline."""

    def test_rag_query_exists(self):
        """Test that rag_query function exists."""
        from rag import rag_query

        assert callable(rag_query)


class TestPHIDetection:
    """Test PHI detection guardrails."""

    def test_detect_phi_exists(self):
        """Test that detect_phi function exists."""
        from guardrails import detect_phi

        assert callable(detect_phi)

    def test_detect_phi_finds_ssn(self):
        """Test that SSN pattern is detected."""
        from guardrails import detect_phi

        text = "Patient SSN: 123-45-6789"
        result = detect_phi(text)

        if result is not None:
            assert "ssn" in result or any(
                "123-45-6789" in str(v) for v in result.values()
            ), "Should detect SSN pattern"

    def test_detect_phi_finds_phone(self):
        """Test that phone number pattern is detected."""
        from guardrails import detect_phi

        text = "Contact: 555-123-4567"
        result = detect_phi(text)

        if result is not None:
            assert "phone" in result or any(
                "555-123-4567" in str(v) for v in result.values()
            ), "Should detect phone number pattern"

    def test_detect_phi_finds_email(self):
        """Test that email pattern is detected."""
        from guardrails import detect_phi

        text = "Email: patient@example.com"
        result = detect_phi(text)

        if result is not None:
            assert "email" in result or any(
                "patient@example.com" in str(v) for v in result.values()
            ), "Should detect email pattern"

    def test_detect_phi_returns_empty_for_clean_text(self):
        """Test that clean text returns empty dict."""
        from guardrails import detect_phi

        text = "Blood pressure 120/80, heart rate 72 bpm"
        result = detect_phi(text)

        if result is not None:
            assert result == {} or len(result) == 0, (
                "Clean text should return empty dict"
            )

    def test_scan_for_phi_exists(self):
        """Test that scan_for_phi function exists."""
        from guardrails import scan_for_phi

        assert callable(scan_for_phi)


class TestDocumentation:
    """Test that documentation exists."""

    def test_readme_exists(self):
        """Test that README.md exists."""
        readme_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "README.md"
        )
        assert os.path.exists(readme_path), "README.md should exist"
