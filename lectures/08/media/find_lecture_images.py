#!/usr/bin/env python3
"""
Interactive script to find and download images for FIXME tags in the lecture document.

This script:
1. Parses the lecture document to find FIXME tags
2. Searches for images based on the descriptions
3. Downloads multiple candidate images for each tag
4. Allows the user to interactively select the best image or skip
5. Updates the FIXME tags in the document
"""

import os
import re
import sys
import argparse
import json
from typing import List, Dict, Tuple, Optional, Any
import webbrowser
import os
from pathlib import Path
from dotenv import load_dotenv
import markdown

# Load environment variables from .env file
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# Import our Google Image Search functionality
from google_image_search import (
    GoogleImageSearch,
    CREDENTIALS_PATH,
    CUSTOM_SEARCH_API_KEY,
)

# Constants
LECTURE_PATH = "lectures/08/lecture_08.md"
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")
DOWNLOADS_DIR = "downloads"
TEST_MODE = False  # Set to True to run tests

# Validate required environment variables
if not SEARCH_ENGINE_ID:
    raise ValueError(
        "SEARCH_ENGINE_ID environment variable is not set. Please check your .env file."
    )
if not CUSTOM_SEARCH_API_KEY:
    raise ValueError(
        "CUSTOM_SEARCH_API_KEY environment variable is not set. Please check your .env file."
    )


def parse_fixme_tags(lecture_path: str) -> List[Dict[str, str]]:
    """
    Parse the lecture document to find FIXME tags for images.

    Args:
        lecture_path: Path to the lecture document

    Returns:
        List of dictionaries with FIXME tag information
    """
    fixme_tags = []

    # Regular expression to match FIXME tags for images
    pattern = r"<!-- #FIXME: Add image: (.*?) lectures/08/media/(.*?) -->"

    try:
        with open(lecture_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Find all FIXME tags
        matches = re.findall(pattern, content)
        line_numbers = []

        # Find line numbers for each match
        for i, line in enumerate(content.splitlines(), 1):
            for match in matches:
                tag_text = f"<!-- #FIXME: Add image: {match[0]} lectures/08/media/{match[1]} -->"
                if tag_text in line:
                    line_numbers.append(i)

        for i, match in enumerate(matches):
            description, target_path = match
            line_number = line_numbers[i] if i < len(line_numbers) else None

            fixme_tags.append({
                "description": description,
                "target_path": f"lectures/08/media/{target_path}",
                "original_tag": f"<!-- #FIXME: Add image: {description} lectures/08/media/{target_path} -->",
                "updated_tag": f"<!-- #FIXME: Added candidate image: {description} lectures/08/media/{target_path} -->",
                "line_number": line_number,
            })

        return fixme_tags

    except Exception as e:
        print(f"Error parsing FIXME tags: {e}")
        return []


def search_images(
    search_client: GoogleImageSearch,
    description: str,
    num_results: int = 5,
    start_index: int = 0,
) -> List[Dict[str, Any]]:
    """
    Search for images based on the description.

    Args:
        search_client: GoogleImageSearch instance
        description: Description of the image to search for
        num_results: Number of results to return
        start_index: Starting index for pagination (0-based)

    Returns:
        List of image search results
    """
    try:
        # Search for images
        print(f"Searching for: {description}")
        results = search_client.search_images(
            description, num_results=num_results, start_index=start_index
        )

        if not results:
            print(f"No images found for: {description}")
            return []

        print(f"Found {len(results)} images")
        return results

    except Exception as e:
        print(f"Error searching for images: {e}")
        return []


def download_candidate_images(
    search_client: GoogleImageSearch,
    results: List[Dict[str, Any]],
    tag_info: Dict[str, str],
    max_downloads: int = None,
) -> List[Dict[str, str]]:
    """
    Download candidate images for a FIXME tag.

    Args:
        search_client: GoogleImageSearch instance
        results: List of image search results
        tag_info: Dictionary with FIXME tag information
        max_downloads: Maximum number of images to download (None for unlimited)

    Returns:
        List of dictionaries with candidate image information
    """
    candidates = []

    # Create a unique subdirectory for this tag's candidates
    target_filename = os.path.basename(tag_info["target_path"])
    target_basename, target_ext = os.path.splitext(target_filename)

    # Create downloads directory if it doesn't exist
    # Use fixme_XX naming convention with 0-based indexing
    tag_index = tag_info.get("index", 0)
    candidates_dir = os.path.join(DOWNLOADS_DIR, f"fixme_{tag_index:02d}")
    os.makedirs(candidates_dir, exist_ok=True)

    # Download each image, up to max_downloads
    for i, result in enumerate(results):
        # Stop if we've reached the maximum number of downloads
        if max_downloads is not None and len(candidates) >= max_downloads:
            break

        image_url = result.get("link")
        if not image_url:
            continue

        # Generate a filename for this candidate
        candidate_filename = f"{target_basename}_candidate_{i + 1}{target_ext}"
        candidate_path = os.path.join(candidates_dir, candidate_filename)

        print(f"Downloading candidate {i + 1}/{len(results)}: {image_url}")

        # Download the image
        if search_client.download_image(
            image_url, candidates_dir, candidate_filename
        ):
            # Add to candidates list
            candidates.append({
                "index": i + 1,
                "url": image_url,
                "path": candidate_path,
                "title": result.get("title", ""),
                "snippet": result.get("snippet", ""),
            })

    return candidates


def create_html_from_markdown(
    markdown_content: str, title: str = "Image Viewer"
) -> str:
    """
    Convert markdown content to HTML with styling for better image viewing.

    Args:
        markdown_content: Markdown content to convert
        title: Title for the HTML page

    Returns:
        HTML content with styling
    """
    # Convert markdown to HTML
    html_content = markdown.markdown(markdown_content, extensions=["tables"])

    # Add styling for better image viewing
    styled_html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        h1 {{
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }}
        h2 {{
            margin-top: 30px;
            border-bottom: 1px solid #eee;
            padding-bottom: 5px;
        }}
        h3 {{
            margin-top: 25px;
            background-color: #f5f5f5;
            padding: 8px;
            border-radius: 4px;
        }}
        img {{
            max-width: 100%;
            max-height: 500px;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
            margin: 10px 0;
            display: block;
        }}
        ul {{
            list-style-type: none;
            padding-left: 0;
        }}
        li {{
            margin-bottom: 8px;
        }}
        code {{
            background-color: #f5f5f5;
            padding: 2px 5px;
            border-radius: 3px;
            font-family: monospace;
        }}
        .candidate-section {{
            margin-bottom: 40px;
            padding: 15px;
            border: 1px solid #eee;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    {html_content}
</body>
</html>
"""
    return styled_html


def create_llm_agent_info_file(
    tag_info: Dict[str, str], candidates: List[Dict[str, str]], output_dir: str
) -> Tuple[str, str]:
    """
    Create information files (markdown and HTML) for LLM agent with context about the tag and candidates.

    Args:
        tag_info: Dictionary with FIXME tag information
        candidates: List of dictionaries with candidate image information
        output_dir: Directory to save the info files

    Returns:
        Tuple of paths to the created markdown and HTML files
    """
    # Create markdown content
    markdown_content = f"# FIXME Tag Information\n\n"
    markdown_content += f"## Tag Details\n"
    markdown_content += f"- **Description:** {tag_info['description']}\n"
    markdown_content += f"- **Target Path:** {tag_info['target_path']}\n"
    markdown_content += f"- **Line Number:** {tag_info['line_number']}\n"
    markdown_content += f"- **Original Tag:** `{tag_info['original_tag']}`\n\n"

    markdown_content += f"## Candidate Images\n\n"
    for candidate in candidates:
        rel_path = os.path.relpath(
            candidate["path"],
            os.path.dirname(os.path.join(output_dir, "tag_info.md")),
        )
        markdown_content += f"### Image {candidate['index']}\n"
        markdown_content += f"- **Path:** {rel_path}\n"
        markdown_content += f"- **Title:** {candidate['title']}\n"
        markdown_content += f"- **URL:** {candidate['url']}\n"
        markdown_content += f"\n![Image {candidate['index']}]({rel_path})\n\n"

    # Write markdown file
    info_file_path = os.path.join(output_dir, "tag_info.md")
    with open(info_file_path, "w") as f:
        f.write(markdown_content)

    # Create and write HTML file
    html_content = create_html_from_markdown(
        markdown_content, f"FIXME Tag: {tag_info['description']}"
    )
    html_file_path = os.path.join(output_dir, "tag_info.html")
    with open(html_file_path, "w") as f:
        f.write(html_content)

    return info_file_path, html_file_path


def select_best_image(
    candidates: List[Dict[str, str]],
    tag_info: Dict[str, str],
    interactive: bool = True,
    llm_agent: bool = False,
) -> Optional[Dict[str, str]]:
    """
    Select the best image from the candidates.

    Args:
        candidates: List of dictionaries with candidate image information
        tag_info: Dictionary with FIXME tag information
        interactive: Whether to interactively prompt the user
        llm_agent: Whether the script is being run by an LLM agent

    Returns:
        Dictionary with selected image information, or None if skipped
    """
    if not candidates:
        print("No candidate images available to select from.")
        return None

    # For LLM agent mode, create a subfolder with info file and don't auto-select
    if llm_agent:
        # Create a subfolder for this tag in the lecture's media/downloads directory
        tag_index = tag_info.get("index", 0)
        safe_desc = "".join(
            c if c.isalnum() else "_" for c in tag_info["description"]
        )

        # Extract lecture number from the target path (e.g., "lectures/08/media/...")
        lecture_match = re.match(r"lectures/(\d+)/", tag_info["target_path"])
        lecture_num = (
            lecture_match.group(1) if lecture_match else "08"
        )  # Default to 08 if not found

        # Create the path directly in the lecture's media/downloads directory
        llm_agent_base_dir = f"lectures/{lecture_num}/media/downloads"
        tag_dir = os.path.join(llm_agent_base_dir, f"fixme_{tag_index:02d}")
        os.makedirs(tag_dir, exist_ok=True)

        # Copy candidate images to the tag directory
        for candidate in candidates:
            filename = os.path.basename(candidate["path"])
            dest_path = os.path.join(tag_dir, filename)
            try:
                import shutil

                shutil.copy2(candidate["path"], dest_path)
                # Update the path in the candidate info
                candidate["path"] = dest_path
            except Exception as e:
                print(f"Error copying image to tag directory: {e}")

        # Create info files (markdown and HTML)
        md_file, html_file = create_llm_agent_info_file(
            tag_info, candidates, tag_dir
        )
        print(f"Created LLM agent info files: {md_file} and {html_file}")

        # Note: We intentionally don't open the browser in LLM-agent mode
        # as it would create too many browser windows automatically

        # Don't auto-select or update the tag in LLM agent mode
        print("LLM agent mode: Not auto-selecting or updating tags.")
        return None

    if not interactive:
        # In non-interactive mode, just return the first candidate
        print(f"Auto-selecting first candidate: {candidates[0]['path']}")
        return candidates[0]

    # Display candidates with detailed information
    print("\nCandidate images:")
    for candidate in candidates:
        print(f"{candidate['index']}. {candidate['path']}")
        print(f"   Title: {candidate['title']}")
        print(f"   URL: {candidate['url']}")

    # For human users, try to open the images in the default image viewer
    for candidate in candidates:
        try:
            # Convert to absolute path
            abs_path = os.path.abspath(candidate["path"])
            print(f"Opening {abs_path} in default viewer...")
            webbrowser.open(f"file://{abs_path}")
        except Exception as e:
            print(f"Error opening image: {e}")

    # Prompt user for selection
    while True:
        choice = input(
            "\nSelect the best image (number), 's' to skip, or 'q' to quit: "
        )

        if choice.lower() == "q":
            print("Quitting...")
            sys.exit(0)

        if choice.lower() == "s":
            print("Skipping this tag.")
            return None

        try:
            index = int(choice)
            selected = next(
                (c for c in candidates if c["index"] == index), None
            )

            if selected:
                return selected
            else:
                print(f"Invalid selection: {choice}. Please try again.")

        except ValueError:
            print(
                f"Invalid input: {choice}. Please enter a number, 's', or 'q'."
            )


def copy_selected_image(
    selected: Dict[str, str], tag_info: Dict[str, str]
) -> bool:
    """
    Copy the selected image to the target path.

    Args:
        selected: Dictionary with selected image information
        tag_info: Dictionary with FIXME tag information

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create target directory if it doesn't exist
        target_dir = os.path.dirname(tag_info["target_path"])
        os.makedirs(target_dir, exist_ok=True)

        # Copy the file
        import shutil

        shutil.copy2(selected["path"], tag_info["target_path"])

        print(f"Copied {selected['path']} to {tag_info['target_path']}")
        return True

    except Exception as e:
        print(f"Error copying image: {e}")
        return False


def update_fixme_tag(
    lecture_path: str, original_tag: str, updated_tag: str
) -> bool:
    """
    Update a FIXME tag in the lecture document.

    Args:
        lecture_path: Path to the lecture document
        original_tag: Original FIXME tag
        updated_tag: Updated FIXME tag

    Returns:
        True if the tag was updated, False otherwise
    """
    try:
        with open(lecture_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Replace the tag
        updated_content = content.replace(original_tag, updated_tag)

        # Check if the tag was replaced
        if content == updated_content:
            return False

        # Write the updated content
        with open(lecture_path, "w", encoding="utf-8") as f:
            f.write(updated_content)

        return True

    except Exception as e:
        print(f"Error updating FIXME tag: {e}")
        return False


def test_script():
    """Run tests for the script."""
    print("Running tests...")

    # Test parsing FIXME tags
    test_content = """
    <!-- #FIXME: Add image: Test image 1. lectures/08/media/test1.png -->
    Some content
    <!-- #FIXME: Add image: Test image 2. lectures/08/media/test2.jpg -->
    """

    test_file = "test_lecture.md"
    with open(test_file, "w") as f:
        f.write(test_content)

    tags = parse_fixme_tags(test_file)
    assert len(tags) == 2, f"Expected 2 tags, got {len(tags)}"
    assert tags[0]["description"] == "Test image 1.", (
        f"Expected 'Test image 1.', got '{tags[0]['description']}'"
    )
    assert tags[1]["target_path"] == "lectures/08/media/test2.jpg", (
        f"Expected 'lectures/08/media/test2.jpg', got '{tags[1]['target_path']}'"
    )

    # Clean up
    os.remove(test_file)

    print("All tests passed!")


def main():
    """Main function to run the script."""
    # Run tests if in test mode
    if TEST_MODE:
        test_script()
        return

    parser = argparse.ArgumentParser(
        description="Find and download images for FIXME tags in lecture documents"
    )
    parser.add_argument(
        "--lecture", default=LECTURE_PATH, help="Path to the lecture document"
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
        "--dry-run",
        action="store_true",
        help="Don't actually download images or update tags",
    )
    parser.add_argument(
        "--tag-index",
        type=int,
        help="Process only the specified tag index (0-based)",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run in non-interactive mode (auto-select first image)",
    )
    parser.add_argument(
        "--llm-agent",
        action="store_true",
        help="Run in LLM agent mode (display images for LLM to see)",
    )
    parser.add_argument(
        "--num-results",
        type=int,
        default=5,
        help="Number of image results to fetch per tag",
    )
    parser.add_argument(
        "--test", action="store_true", help="Run tests and exit"
    )
    parser.add_argument(
        "--html-only",
        action="store_true",
        help="Generate HTML files from existing tag_info.md files without searching for new images",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open generated HTML files in browser (not recommended with many files)",
    )

    args = parser.parse_args()

    # Run tests if requested
    if args.test:
        test_script()
        return

    # Check if search engine ID is set, but allow dry run mode
    if not args.engine_id and not args.dry_run:
        print(
            "Error: SEARCH_ENGINE_ID is not set. Please check your .env file or provide it via --engine-id"
        )
        sys.exit(1)

    # If html-only mode is enabled, just convert existing tag_info.md files to HTML
    if args.html_only:
        print("HTML-only mode: Converting existing tag_info.md files to HTML")
        # Find all tag_info.md files in the downloads directory
        for root, dirs, files in os.walk(DOWNLOADS_DIR):
            for file in files:
                if file == "tag_info.md":
                    md_path = os.path.join(root, file)
                    html_path = os.path.join(root, "tag_info.html")
                    print(f"Converting {md_path} to HTML")

                    try:
                        # Read the markdown content
                        with open(md_path, "r", encoding="utf-8") as f:
                            md_content = f.read()

                        # Convert to HTML
                        html_content = create_html_from_markdown(
                            md_content,
                            f"FIXME Tag Viewer: {os.path.basename(root)}",
                        )

                        # Write the HTML file
                        with open(html_path, "w", encoding="utf-8") as f:
                            f.write(html_content)

                        print(f"Created HTML file: {html_path}")

                        # Only open in browser if explicitly requested
                        if args.open_browser and os.path.exists(html_path):
                            html_file_abs = os.path.abspath(html_path)
                            print(
                                f"Opening HTML viewer in browser: {html_file_abs}"
                            )
                            webbrowser.open(f"file://{html_file_abs}")
                    except Exception as e:
                        print(f"Error converting {md_path} to HTML: {e}")

        print("HTML conversion complete!")
        sys.exit(0)

    # Parse FIXME tags
    fixme_tags = parse_fixme_tags(args.lecture)

    if not fixme_tags:
        print("No FIXME tags found in the lecture document.")
        sys.exit(0)

    print(f"Found {len(fixme_tags)} FIXME tags:")
    for i, tag in enumerate(fixme_tags):
        line_info = (
            f" (line {tag['line_number']})" if tag.get("line_number") else ""
        )
        print(
            f"{i + 1}. {tag['description']}{line_info} -> {tag['target_path']}"
        )

    # Initialize the search client
    # Skip service initialization in dry-run mode to avoid API errors
    search_client = GoogleImageSearch(
        api_key=CUSTOM_SEARCH_API_KEY,
        credentials_path=args.credentials,
        search_engine_id=args.engine_id,
        skip_service_init=args.dry_run,
    )

    # Process tags
    if args.tag_index is not None:
        # Process only the specified tag
        if args.tag_index < 0 or args.tag_index >= len(fixme_tags):
            print(
                f"Error: Tag index {args.tag_index} is out of range (0-{len(fixme_tags) - 1})"
            )
            sys.exit(1)

        tags_to_process = [fixme_tags[args.tag_index]]
    else:
        # Process all tags
        tags_to_process = fixme_tags

    # Create downloads directory
    os.makedirs(DOWNLOADS_DIR, exist_ok=True)

    # Process each tag
    for i, tag in enumerate(tags_to_process):
        # Add index to tag info for reference
        # If processing a specific tag, use the provided tag_index
        if args.tag_index is not None:
            tag["index"] = args.tag_index
        else:
            tag["index"] = i

        print(
            f"\nProcessing tag {i + 1}/{len(tags_to_process)}: {tag['description']}"
        )

        if args.dry_run:
            print("Dry run: Skipping search, download, and tag update")
            continue

        # Search for images - request more results to account for potential download failures
        # Google API allows up to 10 results per request, so we'll request 10 or twice the desired number
        search_num_results = min(10, args.num_results * 2)
        results = search_images(
            search_client, tag["description"], search_num_results
        )

        if not results:
            print(
                f"Skipping tag due to no search results: {tag['description']}"
            )
            continue

        # Download candidate images - limit to the requested number
        candidates = download_candidate_images(
            search_client, results, tag, args.num_results
        )

        # If we didn't get enough candidates and there are more results available,
        # try to download more until we reach the desired number
        if len(candidates) < args.num_results:
            print(
                f"Only downloaded {len(candidates)} of {args.num_results} requested images. Trying more..."
            )
            # Calculate how many more we need
            additional_needed = args.num_results - len(candidates)
            # Request more results if available (up to 50 total from Google API)
            more_results = search_images(
                search_client,
                tag["description"],
                additional_needed
                * 2,  # Request more than needed to account for failures
                start_index=len(results),
            )
            if more_results:
                # Try to download these additional images, but only up to the number still needed
                more_candidates = download_candidate_images(
                    search_client, more_results, tag, additional_needed
                )
                candidates.extend(more_candidates)
                print(f"Downloaded {len(candidates)} total candidate images")

        if not candidates:
            print(
                f"Skipping tag due to no valid candidates: {tag['description']}"
            )
            continue

        # Select the best image
        selected = select_best_image(
            candidates, tag, not args.non_interactive, args.llm_agent
        )

        # In LLM agent mode, we don't auto-select or update tags
        if args.llm_agent:
            continue

        if not selected:
            print(f"Skipping tag (no selection): {tag['description']}")
            continue

        # Copy the selected image to the target path
        if copy_selected_image(selected, tag):
            # Update the FIXME tag
            if update_fixme_tag(
                args.lecture, tag["original_tag"], tag["updated_tag"]
            ):
                print(f"Updated FIXME tag in {args.lecture}")
            else:
                print(f"Failed to update FIXME tag in {args.lecture}")

    print("\nDone!")


if __name__ == "__main__":
    main()
