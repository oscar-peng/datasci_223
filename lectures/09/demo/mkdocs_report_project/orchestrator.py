#!/usr/bin/env python3
"""
Orchestrator script for the Clinical Data Analysis Report.
This script coordinates the generation of report elements (charts, data)
and can optionally build the MkDocs site.
"""

import subprocess
import sys
from pathlib import Path

# Ensure the 'reports' module can be found
# Assumes 'reports' is a subdirectory in the same directory as orchestrator.py
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

try:
    from reports import report_generator
except ImportError as e:
    print(f"Error importing report_generator: {e}")
    print(
        f"Ensure 'reports/report_generator.py' exists and sys.path is correct: {sys.path}"
    )
    sys.exit(1)


def main(build_site=True):
    """
    Main function to orchestrate report generation and site build.
    """
    print("🚀 Starting report orchestration...")

    # Step 1: Generate report elements (charts, data JSONs)
    print("\n tahap 1: Generating report elements...")
    try:
        report_generator.generate_report_elements()
        print("✅ Report elements generated successfully.")
    except Exception as e:
        print(f"❌ Error generating report elements: {e}")
        sys.exit(1)

    # Step 2: Build MkDocs site (optional)
    if build_site:
        print("\n tahap 2: Building MkDocs site...")
        try:
            # Ensure we are in the project root for mkdocs build
            # The mkdocs.yml is expected to be in the current working directory
            # when `mkdocs build` is run.
            result = subprocess.run(
                ["mkdocs", "build"],
                capture_output=True,
                text=True,
                check=True,
                cwd=project_root,
            )
            print("✅ MkDocs site built successfully.")
            if result.stdout:
                print("MkDocs Output:\n", result.stdout)
            if result.stderr:
                print("MkDocs Errors/Warnings:\n", result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error building MkDocs site: {e}")
            print("Stdout:\n", e.stdout)
            print("Stderr:\n", e.stderr)
            sys.exit(1)
        except FileNotFoundError:
            print(
                "❌ Error: 'mkdocs' command not found. Ensure MkDocs is installed and in your PATH."
            )
            sys.exit(1)
    else:
        print("\nSkipping MkDocs site build as per 'build_site=False'.")

    print("\n🎉 Orchestration complete!")


if __name__ == "__main__":
    # To run without building the site: python orchestrator.py --no-build
    should_build = True
    if len(sys.argv) > 1 and sys.argv[1] == "--no-build":
        should_build = False
    main(build_site=should_build)
