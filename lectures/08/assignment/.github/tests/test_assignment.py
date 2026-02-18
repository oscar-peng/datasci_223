"""
Tests for Assignment 8: LLM Applications — RAG & Guardrails

Tests verify output artifacts only — students run the notebook first,
then these tests check the saved results.
"""

import pytest
import json
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "output")


class TestPart1:
    """Test Part 1: PHI Guardrails output."""

    def test_phi_results_exist(self):
        """phi_results.json was created."""
        path = os.path.join(OUTPUT_DIR, "phi_results.json")
        assert os.path.exists(path), (
            "phi_results.json should exist in output/. Run the assignment notebook first."
        )

    def test_phi_results_is_list(self):
        """phi_results.json contains a JSON list."""
        path = os.path.join(OUTPUT_DIR, "phi_results.json")
        if not os.path.exists(path):
            pytest.skip("phi_results.json not found")

        with open(path) as f:
            results = json.load(f)

        assert isinstance(results, list), "phi_results.json should contain a list"

    def test_phi_results_count(self):
        """phi_results.json has 4 items (one per test text)."""
        path = os.path.join(OUTPUT_DIR, "phi_results.json")
        if not os.path.exists(path):
            pytest.skip("phi_results.json not found")

        with open(path) as f:
            results = json.load(f)

        assert len(results) == 4, (
            f"Expected 4 PHI results (one per test text), got {len(results)}"
        )

    def test_phi_result_structure(self):
        """Each result has text_index, has_phi, phi_found, and redacted."""
        path = os.path.join(OUTPUT_DIR, "phi_results.json")
        if not os.path.exists(path):
            pytest.skip("phi_results.json not found")

        with open(path) as f:
            results = json.load(f)

        required = {"text_index", "has_phi", "phi_found", "redacted"}
        for i, result in enumerate(results):
            assert isinstance(result, dict), f"Result {i} should be a dict"
            missing = required - set(result.keys())
            assert not missing, f"Result {i} missing fields: {missing}"

    def test_dirty_texts_have_phi(self):
        """First 3 texts should have PHI detected."""
        path = os.path.join(OUTPUT_DIR, "phi_results.json")
        if not os.path.exists(path):
            pytest.skip("phi_results.json not found")

        with open(path) as f:
            results = json.load(f)

        if len(results) < 4:
            pytest.skip("Not enough results to check")

        for i in range(3):
            assert results[i]["has_phi"] is True, (
                f"Text {i} should have PHI detected (has_phi=True)"
            )

    def test_clean_text_no_phi(self):
        """Fourth text (clean vitals) should have no PHI."""
        path = os.path.join(OUTPUT_DIR, "phi_results.json")
        if not os.path.exists(path):
            pytest.skip("phi_results.json not found")

        with open(path) as f:
            results = json.load(f)

        if len(results) < 4:
            pytest.skip("Not enough results to check")

        assert results[3]["has_phi"] is False, (
            "Text 3 (clean vitals) should have no PHI (has_phi=False)"
        )

    def test_dirty_texts_have_redacted(self):
        """Texts with PHI should have [REDACTED] in the redacted field."""
        path = os.path.join(OUTPUT_DIR, "phi_results.json")
        if not os.path.exists(path):
            pytest.skip("phi_results.json not found")

        with open(path) as f:
            results = json.load(f)

        if len(results) < 3:
            pytest.skip("Not enough results to check")

        for i in range(3):
            redacted = results[i].get("redacted", "")
            assert "[REDACTED]" in redacted, (
                f"Text {i} redacted field should contain '[REDACTED]'"
            )


class TestPart2:
    """Test Part 2: RAG Pipeline output."""

    def test_rag_results_exist(self):
        """rag_results.json was created."""
        path = os.path.join(OUTPUT_DIR, "rag_results.json")
        assert os.path.exists(path), (
            "rag_results.json should exist in output/. Run the assignment notebook first."
        )

    def test_rag_results_is_list(self):
        """rag_results.json contains a JSON list."""
        path = os.path.join(OUTPUT_DIR, "rag_results.json")
        if not os.path.exists(path):
            pytest.skip("rag_results.json not found")

        with open(path) as f:
            results = json.load(f)

        assert isinstance(results, list), "rag_results.json should contain a list"

    def test_rag_results_nonempty(self):
        """At least one RAG result exists."""
        path = os.path.join(OUTPUT_DIR, "rag_results.json")
        if not os.path.exists(path):
            pytest.skip("rag_results.json not found")

        with open(path) as f:
            results = json.load(f)

        assert len(results) >= 1, "Should have at least one RAG result"

    def test_rag_result_structure(self):
        """Each result has answer, sources, and query."""
        path = os.path.join(OUTPUT_DIR, "rag_results.json")
        if not os.path.exists(path):
            pytest.skip("rag_results.json not found")

        with open(path) as f:
            results = json.load(f)

        if len(results) == 0:
            pytest.skip("No RAG results to check")

        required = {"answer", "sources", "query"}
        for i, result in enumerate(results):
            assert isinstance(result, dict), f"Result {i} should be a dict"
            missing = required - set(result.keys())
            assert not missing, f"Result {i} missing fields: {missing}"

    def test_rag_answers_are_strings(self):
        """Answers are non-empty strings."""
        path = os.path.join(OUTPUT_DIR, "rag_results.json")
        if not os.path.exists(path):
            pytest.skip("rag_results.json not found")

        with open(path) as f:
            results = json.load(f)

        for i, result in enumerate(results):
            answer = result.get("answer")
            assert isinstance(answer, str), (
                f"Result {i} answer should be a string"
            )
            assert len(answer.strip()) > 0, (
                f"Result {i} answer should not be empty"
            )
