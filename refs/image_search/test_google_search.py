#!/usr/bin/env python3
"""
Test script for Google Image Search functionality.

This script performs a simple test search using the Google Custom Search API
to verify that the API credentials and search engine ID are working correctly.
"""

import os
import sys
import argparse
from google_image_search import (
    GoogleImageSearch,
    CREDENTIALS_PATH,
    SEARCH_ENGINE_ID,
)


def test_search(
    credentials_path, search_engine_id, query="test", num_results=3
):
    """
    Perform a test search using the Google Custom Search API.

    Args:
        credentials_path: Path to the Google service account credentials JSON file
        search_engine_id: The Custom Search Engine ID
        query: The search query to use for testing
        num_results: Number of results to return

    Returns:
        True if the test was successful, False otherwise
    """
    try:
        print(f"Testing Google Image Search with query: '{query}'")
        print(f"Using credentials: {credentials_path}")
        print(f"Using search engine ID: {search_engine_id}")

        # Initialize the search client
        search_client = GoogleImageSearch(credentials_path, search_engine_id)

        # Perform the search
        results = search_client.search_images(query, num_results=num_results)

        if not results:
            print(
                "No results found. This could be normal for some queries, but might indicate an issue."
            )
            return False

        # Print the results
        print(f"\nFound {len(results)} results:")
        for i, result in enumerate(results):
            print(f"\nResult {i + 1}:")
            print(f"  Title: {result.get('title', 'N/A')}")
            print(f"  Link: {result.get('link', 'N/A')}")
            print(
                f"  Thumbnail: {result.get('image', {}).get('thumbnailLink', 'N/A')}"
            )

        print(
            "\nTest successful! The Google Custom Search API is working correctly."
        )
        return True

    except Exception as e:
        print(f"\nError during test: {e}")
        return False


def main():
    """Main function to run the test script."""
    parser = argparse.ArgumentParser(
        description="Test Google Image Search functionality"
    )
    parser.add_argument(
        "--credentials",
        default=CREDENTIALS_PATH,
        help="Path to credentials JSON file",
    )
    parser.add_argument(
        "--engine-id", default=SEARCH_ENGINE_ID, help="Custom Search Engine ID"
    )
    parser.add_argument(
        "--query",
        default="python programming",
        help="Search query to use for testing",
    )
    parser.add_argument(
        "--num-results", type=int, default=3, help="Number of results to return"
    )

    args = parser.parse_args()

    # Check if search engine ID is set
    if args.engine_id == "YOUR_SEARCH_ENGINE_ID":
        print(
            "Error: You need to set the SEARCH_ENGINE_ID in google_image_search.py or provide it via --engine-id"
        )
        sys.exit(1)

    # Run the test
    success = test_search(
        args.credentials, args.engine_id, args.query, args.num_results
    )

    # Exit with appropriate status code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
