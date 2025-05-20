#!/usr/bin/env python3
"""
Google Image Search Script for Computer Vision Lecture

This script uses the Google Custom Search API to search for images,
download them, and process them. It's designed to be compatible with
MCP and/or LLM Agent tool use.
"""

import os
import sys
import json
import argparse
import requests
import subprocess
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse
from google.oauth2 import service_account
from googleapiclient.discovery import build
import logging
import mimetypes
import os
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# Constants
CREDENTIALS_PATH = os.getenv("CREDENTIALS_PATH")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")
CUSTOM_SEARCH_API_KEY = os.getenv("CUSTOM_SEARCH_API_KEY")
API_SERVICE_NAME = "customsearch"
API_VERSION = "v1"
DOWNLOAD_DIR = "lectures/08/media/downloads"
MEDIA_DIR = "lectures/08/media"

# Validate required environment variables
if not SEARCH_ENGINE_ID:
    raise ValueError(
        "SEARCH_ENGINE_ID environment variable is not set. Please check your .env file."
    )
if not CUSTOM_SEARCH_API_KEY:
    raise ValueError(
        "CUSTOM_SEARCH_API_KEY environment variable is not set. Please check your .env file."
    )


class GoogleImageSearch:
    """Class to handle Google Image Search operations."""

    def __init__(
        self,
        api_key: str = None,
        credentials_path: str = None,
        search_engine_id: str = None,
        skip_service_init: bool = False,
    ):
        """
        Initialize the GoogleImageSearch class.

        Args:
            api_key: The Google Custom Search API key
            credentials_path: Path to the Google service account credentials JSON file (optional)
            search_engine_id: The Custom Search Engine ID
            skip_service_init: If True, skip initializing the service (useful for testing)
        """
        self.api_key = api_key or CUSTOM_SEARCH_API_KEY
        self.credentials_path = credentials_path or CREDENTIALS_PATH
        self.search_engine_id = search_engine_id or SEARCH_ENGINE_ID
        self.service = None

        if not skip_service_init:
            self.service = self._build_service()

    def _build_service(self):
        """Build and return the Google Custom Search service."""
        try:
            # Use API key authentication
            service = build(
                API_SERVICE_NAME,
                API_VERSION,
                developerKey=self.api_key,
                cache_discovery=False,
            )
            return service
        except Exception as e:
            logger.error(f"Error building service: {e}")
            raise

    def search_images(
        self,
        query: str,
        num_results: int = 10,
        image_type: str = None,
        image_size: str = None,
        start_index: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Search for images using Google Custom Search API.

        Args:
            query: The search query
            num_results: Number of results to return (max 10 per request)
            image_type: Type of image (clipart, face, lineart, stock, photo, animated)
            image_size: Size of image (huge, icon, large, medium, small, xlarge, xxlarge)
            start_index: Starting index for pagination (0-based)

        Returns:
            List of image results with metadata
        """
        try:
            # Prepare search parameters
            search_params = {
                "q": query,
                "cx": self.search_engine_id,
                "searchType": "image",
                "num": min(num_results, 10),  # API limit is 10 per request
            }

            # Add start index for pagination if provided
            if start_index > 0:
                search_params["start"] = (
                    start_index + 1
                )  # Google API uses 1-based indexing

            # Add optional parameters if provided
            if image_type:
                search_params["imgType"] = image_type
            if image_size:
                search_params["imgSize"] = image_size

            # Execute the search
            results = self.service.cse().list(**search_params).execute()

            # Extract and return image items
            if "items" in results:
                return results["items"]
            else:
                logger.warning(f"No image results found for query: {query}")
                return []

        except Exception as e:
            logger.error(f"Error searching for images: {e}")
            return []

    def download_image(
        self,
        image_url: str,
        output_path: str,
        filename: str = None,
        verify_ssl: bool = False,
        max_retries: int = 3,
        timeout: int = 30,
    ) -> Optional[str]:
        """
        Download an image from a URL using curl.

        Args:
            image_url: URL of the image to download
            output_path: Directory to save the image
            filename: Filename to use (if None, will be derived from URL)
            verify_ssl: Whether to verify SSL certificates
            max_retries: Maximum number of retry attempts
            timeout: Connection timeout in seconds

        Returns:
            Path to the downloaded image or None if download failed
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)

        # Derive filename from URL if not provided
        if not filename:
            parsed_url = urlparse(image_url)
            filename = os.path.basename(parsed_url.path)
            # If filename is empty or doesn't have an extension, use a default
            if not filename or "." not in filename:
                filename = f"image_{hash(image_url) % 10000}.jpg"

        # Full path to save the image
        full_path = os.path.join(output_path, filename)

        # Try to download with retries
        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Download attempt {attempt + 1}/{max_retries} for {image_url}"
                )

                # Build curl command
                curl_cmd = ["curl", "-L", "-o", full_path]

                # Add timeout
                curl_cmd.extend(["--max-time", str(timeout)])

                # Add SSL verification option
                if not verify_ssl:
                    curl_cmd.append("--insecure")

                # Add retry options
                curl_cmd.extend(["--retry", "3"])

                # Add URL
                curl_cmd.append(image_url)

                # Execute curl command
                logger.info(f"Running curl command: {' '.join(curl_cmd)}")
                result = subprocess.run(
                    curl_cmd,
                    capture_output=True,
                    text=True,
                    check=False,  # Don't raise exception on non-zero exit
                )

                # Check if curl was successful
                if result.returncode != 0:
                    logger.warning(
                        f"Curl failed with code {result.returncode}: {result.stderr}"
                    )
                    if attempt < max_retries - 1:
                        # Wait before retrying
                        import time

                        time.sleep(2)
                        continue
                    else:
                        logger.error(
                            f"Failed to download after {max_retries} attempts"
                        )
                        return None

                # Verify it's a valid image
                if self.validate_image(full_path):
                    logger.info(f"Successfully downloaded image to {full_path}")
                    return full_path
                else:
                    logger.warning(
                        f"Downloaded file is not a valid image: {full_path}"
                    )
                    os.remove(full_path)
                    # Try again if this is not the last attempt
                    if attempt < max_retries - 1:
                        continue
                    return None

            except Exception as e:
                logger.error(f"Error downloading image from {image_url}: {e}")
                if attempt < max_retries - 1:
                    continue
                else:
                    return None

    def validate_image(self, image_path: str) -> bool:
        """
        Validate that a file is a valid image.

        Args:
            image_path: Path to the image file

        Returns:
            True if the file is a valid image, False otherwise
        """
        try:
            # Check if the file exists
            if not os.path.exists(image_path):
                return False

            # Check MIME type first
            mime_type, _ = mimetypes.guess_type(image_path)
            if not mime_type or not mime_type.startswith("image/"):
                return False

            # Try to open the image with PIL
            try:
                with Image.open(image_path) as img:
                    # Verify the image by loading it
                    img.verify()
                    return True
            except Exception:
                return False

        except Exception as e:
            logger.error(f"Error validating image {image_path}: {e}")
            return False

    def search_and_download(
        self,
        query: str,
        output_path: str,
        num_results: int = 5,
        image_type: str = None,
        image_size: str = None,
    ) -> List[str]:
        """
        Search for images and download them using curl.

        Args:
            query: The search query
            output_path: Directory to save the images
            num_results: Number of images to download
            image_type: Type of image
            image_size: Size of image

        Returns:
            List of paths to downloaded images
        """
        # Search for images
        image_results = self.search_images(
            query,
            num_results=num_results,
            image_type=image_type,
            image_size=image_size,
        )

        # Download each image
        downloaded_paths = []
        for i, item in enumerate(image_results):
            image_url = item.get("link")
            if image_url:
                # Generate a filename based on the query and index
                safe_query = "".join(c if c.isalnum() else "_" for c in query)
                filename = f"{safe_query}_{i + 1}.jpg"

                # Download the image with curl (SSL verification disabled by default)
                logger.info(
                    f"Downloading image {i + 1}/{len(image_results)}: {image_url}"
                )
                path = self.download_image(
                    image_url,
                    output_path,
                    filename,
                    verify_ssl=False,
                    max_retries=3,
                    timeout=30,
                )
                if path:
                    downloaded_paths.append(path)
                    logger.info(
                        f"Successfully downloaded image {i + 1} to {path}"
                    )
                else:
                    logger.warning(
                        f"Failed to download image {i + 1} from {image_url}"
                    )

        return downloaded_paths


def main():
    """Main function to run the script from command line."""
    parser = argparse.ArgumentParser(
        description="Google Image Search and Download"
    )
    parser.add_argument("query", help="Search query")
    parser.add_argument(
        "--num",
        type=int,
        default=5,
        help="Number of images to search for and download",
    )
    parser.add_argument(
        "--output",
        default=DOWNLOAD_DIR,
        help="Directory to save downloaded images",
    )
    parser.add_argument(
        "--type",
        choices=["clipart", "face", "lineart", "stock", "photo", "animated"],
        help="Type of image to search for",
    )
    parser.add_argument(
        "--size",
        choices=[
            "huge",
            "icon",
            "large",
            "medium",
            "small",
            "xlarge",
            "xxlarge",
        ],
        help="Size of image to search for",
    )
    parser.add_argument(
        "--credentials",
        default=CREDENTIALS_PATH,
        help="Path to credentials JSON file",
    )
    parser.add_argument(
        "--engine-id", default=SEARCH_ENGINE_ID, help="Custom Search Engine ID"
    )

    args = parser.parse_args()

    # Check if search engine ID is set
    if args.engine_id == "YOUR_SEARCH_ENGINE_ID":
        print(
            "Error: You need to set the SEARCH_ENGINE_ID in the script or provide it via --engine-id"
        )
        sys.exit(1)

    # Initialize the search client
    search_client = GoogleImageSearch(
        api_key=CUSTOM_SEARCH_API_KEY,
        credentials_path=args.credentials,
        search_engine_id=args.engine_id,
    )

    # Search and download images
    downloaded_paths = search_client.search_and_download(
        args.query, args.output, args.num, args.type, args.size
    )

    # Print results
    if downloaded_paths:
        print(f"Successfully downloaded {len(downloaded_paths)} images:")
        for path in downloaded_paths:
            print(f"  - {path}")
    else:
        print("No images were downloaded.")


# MCP Tool Functions
def mcp_search_images(query: str, limit: int = 10) -> Dict[str, Any]:
    """
    MCP tool function to search for images.

    Args:
        query: The search query
        limit: Maximum number of results to return

    Returns:
        Dictionary with search results
    """
    try:
        search_client = GoogleImageSearch()
        results = search_client.search_images(query, num_results=limit)

        return {
            "success": True,
            "results": results,
            "count": len(results),
            "query": query,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "query": query}


def mcp_download_image(
    image_url: str, output_path: str, filename: str
) -> Dict[str, Any]:
    """
    MCP tool function to download an image using curl.

    Args:
        image_url: URL of the image to download
        output_path: Directory to save the image
        filename: Filename to use

    Returns:
        Dictionary with download result
    """
    try:
        search_client = GoogleImageSearch()
        # Use the updated download_image method with curl
        path = search_client.download_image(
            image_url,
            output_path,
            filename,
            verify_ssl=False,  # Default to not verifying SSL to handle problematic URLs
        )

        if path:
            return {"success": True, "path": path, "url": image_url}
        else:
            return {
                "success": False,
                "error": "Failed to download or validate image",
                "url": image_url,
            }
    except Exception as e:
        return {"success": False, "error": str(e), "url": image_url}


def mcp_analyze_images(
    search_results: List[Dict[str, Any]], criteria: str
) -> Dict[str, Any]:
    """
    MCP tool function to analyze image search results based on criteria.

    Args:
        search_results: List of image search results
        criteria: Criteria for selecting the best images

    Returns:
        Dictionary with analysis results
    """
    # This is a placeholder for more sophisticated analysis
    # In a real implementation, this could use image analysis APIs or ML models
    try:
        # Simple filtering based on title and snippet matching criteria
        filtered_results = []
        criteria_terms = criteria.lower().split()

        for result in search_results:
            score = 0
            title = result.get("title", "").lower()
            snippet = result.get("snippet", "").lower()

            # Score based on criteria terms appearing in title and snippet
            for term in criteria_terms:
                if term in title:
                    score += 2
                if term in snippet:
                    score += 1

            # Add score to result
            result["relevance_score"] = score
            filtered_results.append(result)

        # Sort by relevance score
        sorted_results = sorted(
            filtered_results,
            key=lambda x: x.get("relevance_score", 0),
            reverse=True,
        )

        return {
            "success": True,
            "analyzed_results": sorted_results,
            "criteria": criteria,
            "count": len(sorted_results),
        }
    except Exception as e:
        return {"success": False, "error": str(e), "criteria": criteria}


if __name__ == "__main__":
    main()
