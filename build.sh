#!/bin/bash
set -e

# Activate virtual environment
source .venv/bin/activate

# Install Playwright dependencies
playwright install chromium

# Build the site
mkdocs build
