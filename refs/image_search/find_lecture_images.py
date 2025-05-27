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
from pathlib import Path
from dotenv import load_dotenv
import markdown  # Keep for potential future use, though not directly used in HTML generation now
import tempfile

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
LECTURE_PATH = "lectures/08/lecture_08.md"  # Default, can be overridden by args
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")
DOWNLOADS_DIR = "downloads"  # Base for temporary candidate downloads
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
    Parse the lecture document to find FIXME tags for images in the format ![Description](#FIXME).

    Args:
        lecture_path: Path to the lecture document.

    Returns:
        List of dictionaries with FIXME tag information.
    """
    fixme_tags = []
    pattern = r"!\[(.*?)\]\(#FIXME\)"
    tag_idx = 0

    try:
        with open(lecture_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line_num_zero_based, line_content in enumerate(lines):
            for match in re.finditer(pattern, line_content):
                description = match.group(1)
                original_tag_matched_text = match.group(0)

                lecture_doc_path_obj = Path(lecture_path)
                lecture_dir = lecture_doc_path_obj.parent

                safe_desc_part = "".join(
                    c if c.isalnum() else "_"
                    for c in description.lower().replace(" ", "_")
                )[:30]
                image_filename = f"{safe_desc_part}_{tag_idx:02d}.png"
                relative_image_path_for_md = f"media/{image_filename}"
                target_path_for_saving_str = str(
                    lecture_dir / "media" / image_filename
                )
                updated_markdown_link = (
                    f"![{description}]({relative_image_path_for_md})"
                )

                fixme_tags.append({
                    "description": description,
                    "target_path": target_path_for_saving_str,
                    "original_tag": original_tag_matched_text,
                    "updated_tag": updated_markdown_link,
                    "line_number": line_num_zero_based + 1,
                    "relative_image_path_for_md": relative_image_path_for_md,
                    "index": tag_idx,
                })
                tag_idx += 1
        return fixme_tags
    except Exception as e:
        print(f"Error parsing FIXME tags in {lecture_path}: {e}")
        return []


def search_images(
    search_client: GoogleImageSearch,
    description: str,
    num_results: int = 5,
    start_index: int = 0,
) -> List[Dict[str, Any]]:
    """
    Search for images based on the description.
    """
    try:
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
    """
    candidates = []
    target_filename_from_tag = os.path.basename(tag_info["target_path"])
    target_basename, target_ext = os.path.splitext(target_filename_from_tag)
    tag_index = tag_info.get("index", 0)
    candidates_dir = os.path.join(DOWNLOADS_DIR, f"fixme_{tag_index:02d}")
    os.makedirs(candidates_dir, exist_ok=True)

    for i, result in enumerate(results):
        if max_downloads is not None and len(candidates) >= max_downloads:
            break
        image_url = result.get("link")
        if not image_url:
            continue
        candidate_filename = f"{target_basename}_candidate_{i + 1}{target_ext}"
        candidate_path = os.path.join(candidates_dir, candidate_filename)
        print(f"Downloading candidate {i + 1}/{len(results)}: {image_url}")
        if search_client.download_image(
            image_url, candidates_dir, candidate_filename
        ):
            candidates.append({
                "index": i + 1,
                "url": image_url,
                "path": candidate_path,
                "title": result.get("title", ""),
                "snippet": result.get("snippet", ""),
            })
    return candidates


def create_html_from_markdown(
    markdown_content: str,
    title: str = "Image Viewer",
    candidates: List[Dict[str, str]] = None,
) -> str:
    """
    Create a simple HTML page with the first four images tiled in a grid.
    """
    styled_html = f"""<!DOCTYPE html>
<html><head><title>{title}</title><style>
body {{font-family: Arial, sans-serif; margin: 0; padding: 20px; max-width: 1200px; margin: 0 auto;}}
h1 {{text-align: center; margin-bottom: 20px;}}
.image-grid {{display: grid; grid-template-columns: repeat(2, 1fr); grid-gap: 20px;}}
.image-container {{border: 1px solid #ddd; border-radius: 4px; padding: 10px; text-align: center;}}
.image-container img {{max-width: 100%; max-height: 400px; object-fit: contain;}}
.image-label {{font-size: 24px; font-weight: bold; margin-top: 10px;}}
</style></head><body><h1>{title}</h1><div class="image-grid">
"""
    if candidates:
        for i, candidate in enumerate(candidates[:4]):
            rel_path = os.path.basename(candidate["path"])
            styled_html += f"""
        <div class="image-container">
            <img src="{rel_path}" alt="Image {candidate["index"]}">
            <div class="image-label">Image {candidate["index"]}</div>
        </div>"""
    styled_html += """
    </div></body></html>"""
    return styled_html


def create_llm_agent_info_file(
    tag_info: Dict[str, str], candidates: List[Dict[str, str]], output_dir: str
) -> Tuple[str, str]:
    """
    Create simplified information files (markdown and HTML) for LLM agent or human review.
    Images are expected to be copied to output_dir before calling this.
    """
    markdown_content = f"# FIXME Tag Information\n\n## Tag Details\n"
    markdown_content += f"Description: {tag_info['description']}\n"
    markdown_content += (
        f"Target Path (for saving final image): {tag_info['target_path']}\n"
    )
    markdown_content += f"Line Number: {tag_info['line_number']}\n"
    markdown_content += f"Original Tag: `{tag_info['original_tag']}`\n\n"
    markdown_content += f"## Candidate Images (in this folder)\n\n"
    for candidate in candidates:
        rel_path = os.path.basename(candidate["path"])
        markdown_content += f"### Image {candidate['index']}\n"
        markdown_content += f"Filename: {rel_path}\n"
        markdown_content += f"\n![Image {candidate['index']}]({rel_path})\n\n"

    info_file_path = os.path.join(output_dir, "tag_info.md")
    with open(info_file_path, "w") as f:
        f.write(markdown_content)

    html_content = create_html_from_markdown(
        "", f"Review Images for: {tag_info['description']}", candidates
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
    """
    if not candidates:
        print("No candidate images available to select from.")
        return None

    tag_index = tag_info.get("index", 0)
    lecture_media_path = Path(tag_info["target_path"]).parent
    lecture_path_obj = lecture_media_path.parent

    review_folder_name = (
        "downloads_for_llm" if llm_agent else "human_review_candidates"
    )
    review_base_dir = lecture_path_obj / "media" / review_folder_name

    safe_desc_stem = "".join(
        c if c.isalnum() else "_"
        for c in Path(tag_info["target_path"]).stem.rsplit("_", 1)[0]
    )
    tag_review_dir = review_base_dir / f"fixme_{tag_index:02d}_{safe_desc_stem}"
    os.makedirs(tag_review_dir, exist_ok=True)

    copied_candidates_for_review = []
    for candidate in candidates:
        filename = os.path.basename(candidate["path"])
        dest_path = tag_review_dir / filename
        try:
            import shutil

            shutil.copy2(candidate["path"], dest_path)
            copied_candidates_for_review.append({
                **candidate,
                "path": str(dest_path),
            })
        except Exception as e:
            print(
                f"Error copying image to review directory {tag_review_dir}: {e}"
            )

    if not copied_candidates_for_review:
        print("No images were successfully copied for review.")
        return None

    md_file, html_file = create_llm_agent_info_file(
        tag_info, copied_candidates_for_review, str(tag_review_dir)
    )
    print(f"Created review info files: {md_file} and {html_file}")

    if llm_agent:
        print(
            f"LLM agent mode: Review images for description '{tag_info['description']}' in folder: {tag_review_dir}"
        )
        print(
            "Not auto-selecting or updating tags. Manual review required by LLM."
        )
        return None

    if not interactive:
        print(
            f"Non-interactive mode: Auto-selecting first candidate: {candidates[0]['path']}"
        )
        return candidates[0]

    print(f"Generated HTML review page: {html_file}")
    try:
        webbrowser.open(f"file://{Path(html_file).resolve()}")
        print(
            f"Opened HTML review page in browser. Please review and make your selection in the terminal."
        )
    except Exception as e:
        print(
            f"Error opening HTML review page: {e}. Please open it manually: {html_file}"
        )

    print(
        "\nCandidate images (original download paths listed below for reference):"
    )
    for cand in candidates:
        print(f"{cand['index']}. {cand['path']}")

    while True:
        choice = input(
            "\nSelect the best image (number from HTML page/list), 's' to skip, or 'q' to quit: "
        )
        if choice.lower() == "q":
            sys.exit(0)
        if choice.lower() == "s":
            print("Skipping this tag.")
            return None
        try:
            index = int(choice)
            selected_original_candidate = next(
                (c for c in candidates if c["index"] == index), None
            )
            if selected_original_candidate:
                return selected_original_candidate
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
    Copy the selected image (from temp download) to its final target path.
    """
    try:
        final_target_path = Path(tag_info["target_path"])
        target_dir = final_target_path.parent
        os.makedirs(target_dir, exist_ok=True)
        import shutil

        shutil.copy2(selected["path"], str(final_target_path))
        print(f"Copied {selected['path']} to {final_target_path}")
        return True
    except Exception as e:
        print(f"Error copying image: {e}")
        return False


def update_fixme_tag(
    lecture_path: str, original_tag: str, updated_tag: str
) -> bool:
    """
    Update a FIXME tag in the lecture document.
    """
    try:
        with open(lecture_path, "r", encoding="utf-8") as f:
            content = f.read()
        updated_content = content.replace(original_tag, updated_tag)
        if content == updated_content:
            print(
                f"Warning: Tag '{original_tag}' not found or not replaced in {lecture_path}"
            )
            return False
        with open(lecture_path, "w", encoding="utf-8") as f:
            f.write(updated_content)
        return True
    except Exception as e:
        print(f"Error updating FIXME tag: {e}")
        return False


def test_script():
    """Run tests for the script."""
    print("Running tests...")
    test_content = """
    This is a test lecture.
    Here is an image: ![Test image 1 for markdown](#FIXME)
    Some other content.
    And another one: ![Another test image, with, commas and spaces](#FIXME)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        test_lecture_dir = Path(tmpdir) / "lectures" / "00"
        test_lecture_dir.mkdir(parents=True, exist_ok=True)
        test_file_path = test_lecture_dir / "test_lecture.md"
        with open(test_file_path, "w") as f:
            f.write(test_content)
        tags = parse_fixme_tags(str(test_file_path))
        assert len(tags) == 2, f"Expected 2 tags, got {len(tags)}"

        assert tags[0]["description"] == "Test image 1 for markdown"
        expected_filename_0 = "test_image_1_for_markdown_00.png"
        expected_target_path_0 = str(
            test_lecture_dir / "media" / expected_filename_0
        )
        expected_relative_path_0 = f"media/{expected_filename_0}"
        assert tags[0]["target_path"] == expected_target_path_0
        assert tags[0]["original_tag"] == "![Test image 1 for markdown](#FIXME)"
        assert (
            tags[0]["updated_tag"]
            == f"![Test image 1 for markdown]({expected_relative_path_0})"
        )
        assert tags[0]["line_number"] == 3
        assert tags[0]["index"] == 0

        assert (
            tags[1]["description"]
            == "Another test image, with, commas and spaces"
        )
        expected_filename_1 = "another_test_image_with_comm_01.png"
        expected_target_path_1 = str(
            test_lecture_dir / "media" / expected_filename_1
        )
        expected_relative_path_1 = f"media/{expected_filename_1}"
        assert tags[1]["target_path"] == expected_target_path_1
        assert (
            tags[1]["original_tag"]
            == "![Another test image, with, commas and spaces](#FIXME)"
        )
        assert (
            tags[1]["updated_tag"]
            == f"![Another test image, with, commas and spaces]({expected_relative_path_1})"
        )
        assert tags[1]["line_number"] == 5
        assert tags[1]["index"] == 1
    print("All tests passed!")


def main():
    """Main function to run the script."""
    if TEST_MODE:
        test_script()
        return
    parser = argparse.ArgumentParser(
        description="Find/download images for FIXME tags"
    )
    parser.add_argument(
        "--lecture", default=LECTURE_PATH, help="Path to lecture document"
    )
    parser.add_argument(
        "--credentials",
        default=CREDENTIALS_PATH,
        help="Path to credentials JSON",
    )
    parser.add_argument(
        "--engine-id", default=SEARCH_ENGINE_ID, help="Custom Search Engine ID"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="No downloads or tag updates"
    )
    parser.add_argument(
        "--tag-index",
        type=int,
        help="Process only specified tag index (0-based)",
    )
    parser.add_argument(
        "--non-interactive", action="store_true", help="Auto-select first image"
    )
    parser.add_argument(
        "--llm-agent",
        action="store_true",
        help="LLM agent mode (prepare files for review)",
    )
    parser.add_argument(
        "--num-results", type=int, default=4, help="Num image results per tag"
    )
    parser.add_argument(
        "--test", action="store_true", help="Run tests and exit"
    )
    parser.add_argument(
        "--html-only",
        action="store_true",
        help="Generate HTML from existing tag_info.md",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open generated HTML (use with care)",
    )
    args = parser.parse_args()

    if args.test:
        test_script()
        return
    if not args.engine_id and not args.dry_run:
        print("Error: SEARCH_ENGINE_ID not set. Check .env or use --engine-id")
        sys.exit(1)

    if args.html_only:
        print("HTML-only mode: Converting existing review files to HTML")
        # Define lecture_path_obj based on the current lecture argument for html-only mode
        current_lecture_path_obj = Path(args.lecture)
        lecture_folder_for_html_only = (
            current_lecture_path_obj.parent
        )  # e.g. lectures/09

        for review_type_folder in [
            "human_review_candidates",
            "downloads_for_llm",
        ]:
            review_base = (
                lecture_folder_for_html_only / "media" / review_type_folder
            )
            if review_base.exists():
                for tag_specific_folder in review_base.iterdir():
                    if tag_specific_folder.is_dir():
                        md_path = tag_specific_folder / "tag_info.md"
                        html_path = tag_specific_folder / "tag_info.html"
                        if md_path.exists():
                            print(f"Converting {md_path} to HTML")
                            try:
                                with open(md_path, "r", encoding="utf-8") as f:
                                    md_content = f.read()
                                candidates_for_html = []
                                # Correctly list images from the tag_specific_folder for HTML generation
                                for img_file_path in tag_specific_folder.glob(
                                    "*_candidate_*.png"
                                ):  # Or other extensions
                                    candidates_for_html.append({
                                        "index": len(candidates_for_html) + 1,
                                        "path": str(img_file_path.name),
                                    })  # Use only filename for rel path in HTML

                                # Extract description from md_content or folder name if possible for a better title
                                title_desc_match = re.search(
                                    r"Description: (.*?)\n", md_content
                                )
                                page_title = (
                                    f"Review: {tag_specific_folder.name}"
                                )
                                if title_desc_match:
                                    page_title = f"Review Images for: {title_desc_match.group(1)}"

                                html_content = create_html_from_markdown(
                                    "", page_title, candidates_for_html
                                )
                                with open(
                                    html_path, "w", encoding="utf-8"
                                ) as f:
                                    f.write(html_content)
                                print(f"Created HTML: {html_path}")
                                if args.open_browser:
                                    webbrowser.open(
                                        f"file://{html_path.resolve()}"
                                    )
                            except Exception as e:
                                print(f"Error converting {md_path}: {e}")
        print("HTML conversion complete!")
        sys.exit(0)

    fixme_tags = parse_fixme_tags(args.lecture)
    if not fixme_tags:
        print(f"No FIXME tags in {args.lecture}.")
        sys.exit(0)
    print(f"Found {len(fixme_tags)} FIXME tags in {args.lecture}:")
    for i, tag in enumerate(
        fixme_tags
    ):  # Use 'i' from enumerate for display if 'index' isn't what's wanted here
        print(
            f"{tag['index'] + 1}. {tag['description']} (line {tag.get('line_number', 'N/A')}) -> target: {tag['target_path']}"
        )

    search_client = GoogleImageSearch(
        CUSTOM_SEARCH_API_KEY, args.credentials, args.engine_id, args.dry_run
    )

    tags_to_process_indices = range(len(fixme_tags))
    if args.tag_index is not None:
        if 0 <= args.tag_index < len(fixme_tags):
            tags_to_process_indices = [args.tag_index]
        else:
            print(f"Error: Tag index {args.tag_index} out of range.")
            sys.exit(1)

    os.makedirs(DOWNLOADS_DIR, exist_ok=True)

    for current_idx_in_loop, original_list_idx in enumerate(
        tags_to_process_indices
    ):
        tag = fixme_tags[original_list_idx]
        print(
            f"\nProcessing tag {current_idx_in_loop + 1}/{len(tags_to_process_indices)} (Doc index {tag['index']}): {tag['description']}"
        )
        if args.dry_run:
            print("Dry run: Skipping actual processing")
            continue

        search_n = min(10, args.num_results * 2)
        results = search_images(search_client, tag["description"], search_n)
        if not results:
            print(f"No search results for: {tag['description']}")
            continue

        candidates = download_candidate_images(
            search_client, results, tag, args.num_results
        )
        if len(candidates) < args.num_results and len(results) == search_n:
            print(
                f"Need {args.num_results - len(candidates)} more images. Fetching more..."
            )
            more_results = search_images(
                search_client,
                tag["description"],
                (args.num_results - len(candidates)) * 2,
                search_n,
            )
            if more_results:
                candidates.extend(
                    download_candidate_images(
                        search_client,
                        more_results,
                        tag,
                        args.num_results - len(candidates),
                    )
                )

        if not candidates:
            print(f"No valid candidates for: {tag['description']}")
            continue

        selected = select_best_image(
            candidates, tag, not args.non_interactive, args.llm_agent
        )
        if args.llm_agent or not selected:
            continue

        if copy_selected_image(selected, tag):
            if update_fixme_tag(
                args.lecture, tag["original_tag"], tag["updated_tag"]
            ):
                print(f"Updated FIXME tag in {args.lecture}")
            else:
                print(f"Failed to update FIXME tag in {args.lecture}")
    print("\nDone!")


if __name__ == "__main__":
    main()
