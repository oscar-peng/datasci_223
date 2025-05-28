# Demo 3a Guide: Building an Automated Report with MkDocs

**Goal:** Understand and build a complete MkDocs site that automatically generates and embeds interactive charts and diagrams for a clinical data analysis report. This demo uses a self-contained project structure.

**Project Location:** All files for this demo are located in the `lectures/09/demo/mkdocs_report_project/` directory.

## Step 1: Explore the Project Structure

Navigate to the `mkdocs_report_project` directory. You will find the following structure:

```
mkdocs_report_project/
├── .github/
│   └── workflows/
│       └── gh-pages.yml  # GitHub Actions workflow for deployment
├── docs/
│   ├── analysis/
│   │   ├── data_overview.md
│   │   ├── heart_rate_analysis.md
│   │   └── risk_factors.md
│   ├── charts/             # Altair chart JSON files will be generated here
│   ├── media/              # For static images, if any
│   ├── about.md
│   ├── index.md
│   ├── interactive_charts.md
│   └── methodology.md
├── scripts/
│   └── generate_charts.py  # Python script to create Altair charts
├── mkdocs.yml              # MkDocs configuration file
└── requirements.txt        # Python package dependencies
```

**Key Components:**
-   **`mkdocs.yml`**: The main configuration file for your MkDocs site. It defines the site name, theme, navigation, plugins, and Markdown extensions.
-   **`docs/`**: This directory contains all your Markdown content files that will be converted into HTML pages.
    -   `docs/charts/`: This is where the `generate_charts.py` script will save the Altair chart specifications as `.json` files.
-   **`scripts/generate_charts.py`**: A Python script that uses Altair to create visualizations and saves them to `docs/charts/`.
-   **`.github/workflows/gh-pages.yml`**: A GitHub Actions workflow file that automates building the MkDocs site and deploying it to GitHub Pages whenever you push changes to your repository's main branch.
-   **`requirements.txt`**: Lists the Python packages needed to run the chart generation script and build the MkDocs site.

## Step 2: Review `mkdocs.yml`

Open `mkdocs_report_project/mkdocs.yml`. Key sections to note:
-   `site_name`, `site_description`, `site_author`, `site_url`: Basic site metadata. **Important:** You'll need to change `site_url` and the `repo_url`/`repo_name` if you plan to deploy this to your own GitHub Pages.
-   `theme`: Configures the Material for MkDocs theme and its features (navigation, color palette).
-   `plugins`:
    -   `search`: Enables the built-in search functionality.
    -   `charts`: Enables the `mkdocs-charts-plugin` for rendering Vega-Lite JSON specifications (which Altair charts are saved as).
-   `markdown_extensions`: Enables various Markdown features like admonitions, Mermaid diagrams (via `pymdownx.superfences`), tabs, etc.
-   `nav`: Defines the navigation structure of your site (sidebar and tabs).

## Step 3: Examine the Chart Generation Script

Open `mkdocs_report_project/scripts/generate_charts.py`. This script:
1.  Imports necessary libraries (`altair`, `pandas`, `numpy`, `pathlib`, `json`).
2.  Defines functions to create sample clinical data (`create_sample_data`).
3.  Defines functions to generate specific Altair charts (e.g., `create_overview_chart`, `create_condition_summary`, etc.). Each chart is saved as a `.json` file in the `docs/charts/` directory.
    -   Notice the `output_dir` path is constructed to be `docs/charts/` relative to the main project directory.
4.  The `main()` function orchestrates data generation and chart creation.

## Step 4: Review Report Content (Markdown Files)

Explore the `.md` files in the `mkdocs_report_project/docs/` directory (e.g., `index.md`, `analysis/heart_rate_analysis.md`).
-   **Embedding Altair Charts:** Charts are embedded using the `vegalite` fenced code block, referencing the JSON files saved by the Python script:
    ```markdown
    ```vegalite
    {
      "schema-url": "charts/overview_scatter.json"
    }
    ```
-   **Embedding Mermaid Diagrams:** Mermaid diagrams are directly written in Markdown:
    ```markdown
    ```mermaid
    graph TD;
        A --> B;
    ```
-   **Other Markdown Features:** Notice the use of admonitions (`!!! info`), tabs (`=== "Tab Title"`), and other features enabled in `mkdocs.yml`.

## Step 5: Set Up Environment and Generate Charts

1.  **Navigate to the project directory:**
    ```bash
    cd lectures/09/demo/mkdocs_report_project
    ```
2.  **Create a virtual environment (recommended) and activate it:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the chart generation script:**
    This script will create the `.json` chart files in `docs/charts/`.
    ```bash
    python scripts/generate_charts.py
    ```
    You should see output indicating charts are being generated and saved.

## Step 6: Build and Serve the MkDocs Site Locally

1.  **Build the site:**
    This command compiles your Markdown files and other assets into a static HTML site in a new `site/` directory within `mkdocs_report_project/`.
    ```bash
    mkdocs build
    ```
2.  **Serve the site locally:**
    This starts a local development server, and you can view your report in a web browser. It usually auto-reloads when you save changes to your Markdown files or `mkdocs.yml`.
    ```bash
    mkdocs serve
    ```
    Open your browser and go to `http://127.0.0.1:8000` (or the address shown in your terminal).

## Step 7: Understanding GitHub Actions for Deployment (Conceptual)

Open `mkdocs_report_project/.github/workflows/gh-pages.yml`. This file defines an automated workflow:
-   **Trigger:** It runs automatically when you `push` changes to the `main` (or `master`) branch of your GitHub repository.
-   **Jobs:**
    -   `checkout code`: Downloads your repository's code.
    -   `Set up Python`: Installs Python.
    -   `Install dependencies`: Installs packages from `requirements.txt`.
    -   `Generate Charts`: Runs your `scripts/generate_charts.py`.
    -   `Build MkDocs site`: Runs `mkdocs build`.
    -   `Deploy to GitHub Pages`: Uses the `peaceiris/actions-gh-pages` action to push the contents of the `site/` directory to a special `gh-pages` branch in your repository. GitHub Pages then serves your site from this branch.

To use this for your own project on GitHub:
1.  Create a new public GitHub repository.
2.  Push the contents of the `mkdocs_report_project/` directory to this repository.
3.  Ensure your repository settings are configured to serve GitHub Pages from the `gh-pages` branch (this is usually automatic after the first successful action run).
4.  Update `site_url`, `repo_name`, and `repo_url` in `mkdocs.yml` to match your repository.

## Success Validation

-   ✅ You can successfully run `python scripts/generate_charts.py` and see `.json` files created in `docs/charts/`.
-   ✅ You can successfully run `mkdocs serve` and view the report at `http://127.0.0.1:8000`.
-   ✅ The report displays text, embedded Altair charts, and Mermaid diagrams correctly.
-   ✅ Navigation links work as expected.
-   ✅ (Optional) If deployed to GitHub Pages, the site is accessible online.

This demo provides a robust template for creating and automating data science reports.