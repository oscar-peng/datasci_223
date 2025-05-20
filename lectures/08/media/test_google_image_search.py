#!/usr/bin/env python3
"""
Pytest-based tests for Google Image Search functionality.
"""

import os
import tempfile
import pytest
from google_image_search import (
    GoogleImageSearch,
    CREDENTIALS_PATH,
    SEARCH_ENGINE_ID,
)

# Skip tests if credentials or search engine ID are not set
skip_if_no_credentials = pytest.mark.skipif(
    not os.path.exists(CREDENTIALS_PATH),
    reason="Google API credentials file not found",
)

skip_if_no_search_engine = pytest.mark.skipif(
    SEARCH_ENGINE_ID == "YOUR_SEARCH_ENGINE_ID",
    reason="Search engine ID not configured",
)


@skip_if_no_credentials
@skip_if_no_search_engine
def test_search_images():
    """Test that search_images returns results."""
    search_client = GoogleImageSearch(CREDENTIALS_PATH, SEARCH_ENGINE_ID)
    results = search_client.search_images("python programming", num_results=3)

    assert results is not None
    assert isinstance(results, list)
    assert len(results) > 0


@skip_if_no_credentials
@skip_if_no_search_engine
def test_image_result_structure():
    """Test that image search results have the expected structure."""
    search_client = GoogleImageSearch(CREDENTIALS_PATH, SEARCH_ENGINE_ID)
    results = search_client.search_images("python programming", num_results=1)

    if not results:
        pytest.skip("No search results returned")

    result = results[0]
    assert "link" in result, "Result should have a 'link' field"
    assert "title" in result, "Result should have a 'title' field"


@skip_if_no_credentials
@skip_if_no_search_engine
def test_download_image(tmp_path):
    """Test downloading an image to a temporary directory."""
    search_client = GoogleImageSearch(CREDENTIALS_PATH, SEARCH_ENGINE_ID)
    results = search_client.search_images("python logo", num_results=1)

    if not results:
        pytest.skip("No search results returned")

    image_url = results[0].get("link")
    if not image_url:
        pytest.skip("No image URL in search result")

    # Create a temporary directory for the test
    download_dir = tmp_path / "downloads"
    download_dir.mkdir()

    # Download the image
    result = search_client.download_image(
        image_url, str(download_dir), "test_image.png"
    )

    assert result, "Image download should succeed"
    assert (download_dir / "test_image.png").exists(), (
        "Downloaded image file should exist"
    )


# Tests that don't require Google API credentials
def test_validate_image_with_nonexistent_file():
    """Test that validate_image returns False for a nonexistent file."""
    # Use dummy credentials for testing validation functionality
    search_client = GoogleImageSearch(
        "dummy_credentials.json",
        "dummy_search_engine_id",
        skip_service_init=True,
    )
    result = search_client.validate_image("nonexistent_file.jpg")
    assert result is False


def test_validate_image_with_text_file():
    """Test that validate_image returns False for a text file."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
        temp_file.write(b"This is not an image")
        temp_path = temp_file.name

    try:
        # Use dummy credentials for testing validation functionality
        search_client = GoogleImageSearch(
            "dummy_credentials.json",
            "dummy_search_engine_id",
            skip_service_init=True,
        )
        result = search_client.validate_image(temp_path)
        assert result is False
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    # This allows running the tests directly with python
    # (though pytest is the recommended way)
    pytest.main(["-v", __file__])
