"""
Tests for LLM Prompt Engineering Assignment

These tests verify that:
1. Required functions exist
2. Prompt building works correctly
3. Response validation works
4. JSON parsing handles various formats
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class TestPromptBuilding:
    """Test prompt construction functions."""

    def test_build_prompt_exists(self):
        """Test that build_prompt function exists."""
        from extractor import build_prompt

        assert callable(build_prompt)

    def test_build_prompt_returns_string(self):
        """Test that build_prompt returns a string."""
        from extractor import build_prompt

        result = build_prompt("Sample clinical note text")

        assert result is not None, (
            "build_prompt should return a string, not None"
        )
        assert isinstance(result, str), "build_prompt should return a string"

    def test_build_prompt_includes_note(self):
        """Test that the prompt includes the clinical note."""
        from extractor import build_prompt

        note = "Patient has diabetes with glucose 250 mg/dL"
        result = build_prompt(note)

        if result:
            assert note in result or "diabetes" in result.lower(), (
                "Prompt should include the clinical note content"
            )


class TestValidation:
    """Test response validation functions."""

    def test_validate_response_exists(self):
        """Test that validate_response function exists."""
        from extractor import validate_response

        assert callable(validate_response)

    def test_validate_response_accepts_valid(self):
        """Test that validation accepts valid responses."""
        from extractor import validate_response

        valid_response = {
            "diagnosis": "Type 2 Diabetes",
            "medications": ["metformin", "lisinopril"],
            "lab_values": {"glucose": "250 mg/dL"},
            "confidence": 0.85,
        }

        result = validate_response(valid_response)

        assert result is not None, "validate_response should return a boolean"
        assert result is True, "Valid response should pass validation"

    def test_validate_response_rejects_missing_fields(self):
        """Test that validation rejects responses with missing fields."""
        from extractor import validate_response

        invalid_response = {
            "diagnosis": "Type 2 Diabetes"
            # Missing: medications, lab_values, confidence
        }

        result = validate_response(invalid_response)

        assert result is False, (
            "Response with missing fields should fail validation"
        )


class TestJSONParsing:
    """Test JSON parsing functions."""

    def test_parse_json_response_exists(self):
        """Test that parse_json_response function exists."""
        from extractor import parse_json_response

        assert callable(parse_json_response)

    def test_parse_json_response_handles_clean_json(self):
        """Test parsing clean JSON."""
        from extractor import parse_json_response

        response_text = (
            '{"diagnosis": "Pneumonia", "medications": ["azithromycin"]}'
        )
        result = parse_json_response(response_text)

        assert result is not None, "Should parse clean JSON"
        assert result.get("diagnosis") == "Pneumonia"

    def test_parse_json_response_handles_markdown_wrapped(self):
        """Test parsing JSON wrapped in markdown code blocks."""
        from extractor import parse_json_response

        response_text = """Here is the extracted data:

```json
{"diagnosis": "Pneumonia", "medications": ["azithromycin"]}
```

This is my analysis."""

        result = parse_json_response(response_text)

        # Should either parse successfully or return None gracefully
        if result is not None:
            assert result.get("diagnosis") == "Pneumonia"

    def test_parse_json_response_handles_invalid(self):
        """Test that invalid JSON returns None."""
        from extractor import parse_json_response

        response_text = "This is not JSON at all"
        result = parse_json_response(response_text)

        assert result is None, "Invalid JSON should return None"


class TestExtraction:
    """Test the main extraction function."""

    def test_extract_entities_exists(self):
        """Test that extract_entities function exists."""
        from extractor import extract_entities

        assert callable(extract_entities)


class TestDocumentation:
    """Test that documentation exists."""

    def test_readme_exists(self):
        """Test that README.md exists."""
        readme_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "README.md"
        )
        assert os.path.exists(readme_path), "README.md should exist"
