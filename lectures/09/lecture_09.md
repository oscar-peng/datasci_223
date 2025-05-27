# Lecture 09: Data Visualization, Diagramming, Reporting & Dashboards

**Overall Goal:** Equip students with skills to create insightful diagrams (Mermaid), interactive data visualizations (Altair), automated shareable reports (MkDocs), and simple interactive dashboards (Streamlit), applied to health data contexts, culminating in engaging data storytelling examples.

**Target Audience:** Health data science master's students (beginners in programming).
**Lecture Duration:** 90 minutes.
**Format:** Long-form Markdown.

---

## 0. Introduction (5 minutes)

A common challenge in data science is transforming complex health data into clear, actionable insights. Effective communication is as crucial as the analysis itself. Consider the difference between presenting a dense table of numbers versus an interactive chart or a simple dashboard that highlights key findings.
![Data communication dense table vs interactive dashboard comparison](media/data_communication_dense_table_00.png)

**Lecture Objectives:**
*   Create diagrams as code using Mermaid for clear process documentation.
*   Craft interactive visualizations with Altair, including building towards a Gapminder-style dynamic chart.
*   Build automated, shareable reports using MkDocs for disseminating findings.
*   Develop simple, practical dashboards and then a more dynamic one with Streamlit for data exploration.
*   Apply these tools to health data scenarios, focusing on principles of effective communication.

**Agenda Overview:**

```mermaid
graph TD
    A[Intro: The Power of Visual Communication] --> B(Diagramming: Mermaid);
    B --> C(Interactive Viz: Altair);
    C --> D(Automated Reports: MkDocs);
    D --> E(Dashboards: Streamlit);
```

<!---
*   This lecture builds upon previous sessions on data analysis and machine learning, focusing on how to make the results of such work understandable and impactful.
*   The core theme is moving from raw data and complex analyses to clear narratives and interactive explorations.
*   Effective communication can significantly amplify the value of data science work.
--->

---

## 1. Diagramming as Code with Mermaid (15 minutes)

Visualizing processes, architectures, and workflows is essential for understanding and communicating complex systems, especially in data science. While many tools exist for creating diagrams, the "diagrams as code" approach offers unique advantages for technical projects.

### 1.1. Why "Diagrams as Code"?

*   **Concept:** Treating diagrams as source code offers several advantages. These diagrams are defined using text, making them version-controllable with tools like Git, inherently reproducible, and easier to update systematically.
    <!---
    *   Instead of using a graphical user interface (GUI) to draw shapes and connectors, you write text-based definitions that a tool then renders into a visual diagram.
    *   This is analogous to writing code for software rather than using a WYSIWYG website builder for all web development tasks.
    --->
*   **Benefits:** This approach promotes:
    *   **Consistency:** Diagrams maintain a uniform style, especially across a team or project.
    *   **Version Control:** Changes to diagrams can be tracked, diffed, and reverted using Git, just like any other code. This is invaluable for collaborative projects and understanding the evolution of a design.
        ![Git diff mermaid diagram version control example screenshot](media/git_diff_mermaid_diagram_versi_01.png)
    *   **Reproducibility:** Anyone with the text definition can regenerate the exact same diagram.
    *   **Easy Integration:** Text-based diagrams can be easily embedded into documentation (like MkDocs sites, which we'll cover later), README files, or even code comments.
    *   **Collaboration:** Team members can collaborate on diagrams using familiar code review workflows.
    *   **Accessibility:** Text-based definitions can be more accessible to individuals using screen readers than complex image files, although the rendered output's accessibility also matters.
    <!---
    *   Reproducibility ensures that the diagram accurately reflects the documented system at any point in time.
    *   Ease of integration means diagrams live alongside the documentation or code they describe, reducing the chance of them becoming outdated or lost.
    --->
*   **Contrast with GUI Tools:** GUI-based diagramming tools (e.g., Microsoft Visio, Lucidchart, draw.io) offer a visual interface for drawing. While often user-friendly for initial creation, they can be challenging for:
    *   **Versioning:** Tracking precise changes can be difficult.
    *   **Reproducibility:** Ensuring identical regeneration by different users or on different systems can be tricky.
    *   **Programmatic Updates:** Making systematic changes across many diagrams is often manual.
    *   **Integration with Code/Docs:** Often involves exporting static images, which can become outdated.
    <!---
    *   GUI tools excel at free-form drawing and quick mockups.
    *   "Diagrams as code" tools shine when diagrams need to be maintained, versioned, and integrated with technical documentation over time.
    --->

### 1.2. Introduction to Mermaid

Mermaid is a popular JavaScript-based tool that takes Markdown-inspired text definitions and renders them as diagrams. It's designed to be simple to learn yet powerful enough for a variety of diagramming needs.

*   **What is Mermaid?** Mermaid is a JavaScript-based diagramming and charting tool that uses Markdown-inspired text definitions to dynamically create and modify diagrams. You write text, Mermaid draws the picture.
    <!---
    *   The "Markdown-inspired" part means its syntax is generally human-readable and relatively simple, much like Markdown for text formatting.
    --->
*   **Common Diagram Types:** It supports various diagram types, including:
    *   **Flowcharts:** For visualizing processes, workflows, and decision trees. (e.g., `graph TD; A-->B;`)
        ![Simple fun mermaid flowchart example rendered diagram technical](media/simple_fun_mermaid_flowchart_e_00.png)
    *   **Sequence Diagrams:** For showing interactions between different components or actors over time. (e.g., `sequenceDiagram; Alice->>John: Hello John;`)
        ![Simple mermaid sequence diagram](media/mermaid_sequence.png)
    *   **Gantt Charts:** For project scheduling and tracking task timelines.
    *   **Class Diagrams:** For visualizing object-oriented software structures.
    *   **Entity Relationship Diagrams (ERDs):** For database schema design.
    *   And more (User Journey, Pie Chart, Requirement Diagram, etc.).
    <!---
    *   Flowcharts and sequence diagrams are particularly useful in data science for documenting data pipelines, model workflows, or system interactions.
    --->
*   **Tools for Mermaid:**
    *   **Online Editor:** The [Mermaid Live Editor](https://mermaid.live) is an excellent resource for quickly writing, previewing, and sharing Mermaid diagrams.
    *   **VS Code Extensions:** Many extensions provide live preview capabilities for Mermaid diagrams within Markdown files (e.g., "Markdown Preview Mermaid Support," "Mermaid Markdown Syntax Highlighting").
    *   **MkDocs Integration:** Many MkDocs themes (like Material for MkDocs) have built-in support for Mermaid, or it can be added via plugins. We'll see this later.
    *   **Other Platforms:** GitHub, GitLab, and some other platforms also render Mermaid diagrams directly in Markdown files.
    <!---
    *   The Mermaid Live Editor (mermaid.live) is a convenient online resource for quickly drafting and testing Mermaid diagrams.
    *   Native rendering support in platforms like GitHub and GitLab makes it easy to include diagrams directly in project READMEs or wikis.
    --->

### 1.3. Basic Mermaid Syntax & Examples

Let's focus on flowcharts, as they are broadly applicable.

#### Flowcharts
Flowcharts are used to represent a process, workflow, or algorithm, showing steps as boxes of various kinds, and their order by connecting them with arrows.

*   **Concept:** Visualizing processes, step-by-step logic, and decision points.
*   **Reference Card: Mermaid Flowchart**
    *   **Declaration:** Start with `graph TD;` (for Top-Down) or `graph LR;` (for Left-Right). Other orientations like `BT` (Bottom-Top) and `RL` (Right-Left) also exist.
        <!---
        *   `TD` or `TB` for Top to Bottom.
        *   `LR` for Left to Right.
        --->
    *   **Nodes (Shapes):**
        *   `id[Text]`  Default rectangle: `A[Hard edge]`
        *   `id(Text)`  Rounded rectangle: `B(Round edge)`
        *   `id((Text))` Circle: `C((Circle))`
        *   `id{Text}`  Diamond (for decisions): `D{Decision?}`
        *   `id>Text]`  Asymmetric/Stadium: `E>Stadium]`
        *   Many other shapes are available (parallelogram, trapezoid, etc.).
        <!---
        *   The `id` is a unique identifier for the node, used for linking. The `Text` is what's displayed.
        *   Choosing the right shape can help convey the meaning of a step (e.g., diamond for decisions).
        --->
    *   **Links (Connections):**
        *   `A --> B` (Arrow link from A to B)
        *   `A --- B` (Line link from A to B)
        *   `A -- Text --> B` (Arrow link with text on the arrow)
        *   `A -.-> B` (Dotted arrow link)
        *   `A == Text ==> B` (Thick arrow link with text)
        <!---
        *   Links define the flow and relationships between steps.
        *   Text on links can clarify conditions or actions.
        --->
*   **Minimal Example (Health Data Analysis Pipeline):**
    This diagram outlines a typical workflow for a health data analysis project.
    ```mermaid
    graph TD;
        A[Load PhysioNet Data] --> B(Data Cleaning & Preprocessing);
        B --> C{Select Analysis Type};
        C -- Descriptive Stats --> D[Generate Summary Tables];
        C -- Predictive Model --> E[Train & Evaluate Model];
        D --> F[Visualize Key Metrics];
        E --> F;
        F --> G[Compile Report/Dashboard];
    ```
    <!---
    *   `A[Load PhysioNet Data]`: Represents the initial step of data ingestion.
    *   `B(Data Cleaning & Preprocessing)`: A process step with rounded edges.
    *   `C{Select Analysis Type}`: A decision point, indicated by the diamond shape.
    *   The arrows (`-->`) show the direction of flow.
    *   Text on arrows (`-- Descriptive Stats -->`) clarifies the path taken from a decision.
    --->

### Demo 1: Mermaid Flowchart
*   (Refer to [`lectures/09/demo/01_mermaid_flowchart.md`](lectures/09/demo/01_mermaid_flowchart.md))
    <!---
    *   The first demo will provide hands-on practice with creating a simple flowchart.
    *   Students will apply the syntax learned to visualize a familiar process.
    --->

---

## 2. Interactive Data Visualization with Altair (25 minutes)

While static charts are useful, interactive visualizations empower users to explore data more deeply, uncover patterns, and gain personalized insights. Altair is a Python library that excels at creating a wide range of interactive statistical visualizations with a concise and intuitive syntax.

### 2.1. Beyond Static: The Power of Interaction
*   **Why Interactive?** Interactive visualizations allow users to explore data dynamically through features like tooltips, zooming, panning, and selections. This enhances engagement, facilitates the understanding of complex datasets, and enables users to ask their own questions of the data.
    <!---
    *   Interactivity transforms the audience from passive viewers into active data explorers.
    *   For example, hovering over a data point to see detailed information (tooltip), or selecting a subset of data to see it highlighted in other linked charts.
    --->
    ![Static chart vs interactive chart with tooltips data visualization comparison](media/static_chart_vs_interactive_ch_04.png)

### 2.2. Introduction to Altair
*   **What is Altair?** Altair is a declarative statistical visualization library for Python, built on top of Vega-Lite. "Declarative" means you specify *what* you want to visualize (the mapping from data to visual properties), rather than detailing *how* to draw it step-by-step (imperative).
    <!---
    *   Vega-Lite is a high-level visualization grammar, and Altair provides a Python API to generate Vega-Lite JSON specifications. These JSON specs are then rendered by JavaScript libraries in environments like Jupyter notebooks, web browsers, or MkDocs sites.
    --->
*   **Key Principles (Grammar of Graphics):** Altair follows the Grammar of Graphics, a formal system for describing statistical graphics. Visualizations are built by mapping data columns to visual properties (encodings) of geometric shapes (marks). The core components are:
    *   **Data:** The dataset, typically a Pandas DataFrame. Altair works best with data in a "tidy" long-form format.
    *   **Mark:** The geometric object representing data (e.g., `mark_point()`, `mark_bar()`, `mark_line()`, `mark_area()`, `mark_rect()`).
    *   **Encoding:** The mapping of data fields (columns) to visual channels like:
        *   `x`: x-axis position
        *   `y`: y-axis position
        *   `color`: mark color
        *   `size`: mark size
        *   `shape`: mark shape
        *   `opacity`: mark transparency
        *   `tooltip`: information to show on hover
    <!---
    *   The Grammar of Graphics provides a structured way to think about and construct visualizations, promoting consistency and expressiveness.
    *   Tidy data means each variable forms a column, each observation forms a row, and each type of observational unit forms a table.
    --->
*   **Benefits:** This approach leads to:
    *   **Concise Code:** Complex charts can often be expressed in just a few lines of Python.
    *   **Aesthetically Pleasing Defaults:** Altair charts generally look good out-of-the-box.
    *   **Powerful Interactivity:** Built-in support for selections, tooltips, panning, and zooming.
*   **Comparison (Briefly):**
    *   `plotnine` is another Python library based on the Grammar of Graphics (an implementation of R's `ggplot2`). It shares the declarative philosophy with Altair.
    *   Both Altair and `plotnine` contrast with the more imperative (step-by-step drawing commands) approach of basic `matplotlib`. While `matplotlib` is highly flexible and powerful, creating complex, publication-quality charts can require more verbose code.
    <!---
    *   The choice between Altair and plotnine can depend on familiarity with ggplot2 syntax (for plotnine users) or preference for Vega-Lite's interactivity and web-native output (for Altair users).
    --->

### 2.3. Basic Altair: Building Blocks
Let's look at the fundamental components for creating an Altair chart.

*   **Reference Card: `altair.Chart`**
    *   **Core Object:** `alt.Chart(data)`: This is the starting point. You pass your Pandas DataFrame to it.
        <!---
        *   `alt` is the conventional alias for `import altair as alt`.
        --->
    *   **Mark Type:** `.mark_type()`: Specifies the geometric shape. Examples:
        *   `mark_point()`: For scatter plots.
        *   `mark_bar()`: For bar charts.
        *   `mark_line()`: For line charts.
        *   `mark_area()`: For area charts.
        *   `mark_rect()`: For heatmaps.
    *   **Encodings:** `.encode(...)`: This is where you map data columns to visual properties.
        *   Syntax: `channel='column_name:type_shorthand'`
        *   **Type Shorthands:**
            *   `:Q` - Quantitative (continuous numerical data)
            *   `:N` - Nominal (discrete, unordered categorical data)
            *   `:O` - Ordinal (discrete, ordered categorical data)
            *   `:T` - Temporal (date/time data)
        *   Example: `alt.X('age:Q')`, `alt.Y('systolic_bp:Q')`, `alt.Color('gender:N')`
        <!---
        *   Specifying the correct data type is crucial for Altair to apply appropriate scales, axes, and legends.
        --->
    *   **Properties:** `.properties(...)`: To set overall chart attributes.
        *   `width=W` (integer, pixels)
        *   `height=H` (integer, pixels)
        *   `title='My Chart Title'`
    *   **Interactivity:** `.interactive()`: A convenient shortcut to enable basic panning and zooming.
    *   **Saving Charts:** `.save('filename.ext')`
        *   `'chart.html'`: Saves as a self-contained HTML file.
        *   `'chart.json'`: Saves the Vega-Lite JSON specification. This is very useful for embedding in web pages or using with tools like MkDocs and Streamlit.
        *   `'chart.png'` or `'chart.svg'`: Saves as a static image. Requires the `vl-convert` package (`pip install vl-convert-python`).
        <!---
        *   `altair_viewer` is another package that can help display charts during development, especially outside of Jupyter environments.
        --->
*   **Minimal Example (Scatter Plot from PhysioNet data snippet):**
    Let's assume we have a Pandas DataFrame `physio_df` from a PhysioNet source with columns like `age`, `heart_rate`, and `patient_id`.
    ```python
    import altair as alt
    import pandas as pd

    # Example: Create a placeholder DataFrame if physio_df is not loaded
    # This is just for demonstration if you run this code block standalone.
    # In a real scenario, physio_df would be loaded from a CSV or other source.
    if 'physio_df' not in locals():
        physio_df = pd.DataFrame({
           'age': [65, 70, 55, 80, 62, 75, 58, 72], 
           'heart_rate': [75, 88, 60, 92, 70, 85, 65, 90], 
           'patient_id': ['P001', 'P002', 'P003', 'P004', 'P005', 'P006', 'P007', 'P008'],
           'gender': ['Male', 'Female', 'Male', 'Female', 'Female', 'Male', 'Male', 'Female']
        })
    
    scatter_plot = alt.Chart(physio_df).mark_point(size=100).encode(
        x='age:Q',  # Age on x-axis, quantitative
        y='heart_rate:Q',  # Heart rate on y-axis, quantitative
        color='gender:N', # Color points by gender (nominal)
        tooltip=['patient_id:N', 'age:Q', 'heart_rate:Q', 'gender:N'] # Info on hover
    ).properties(
        title='Age vs. Heart Rate by Gender'
    ).interactive() # Enable pan and zoom

    # To display in a Jupyter Notebook, this is often enough:
    # scatter_plot 
    
    # To save (uncomment the one you need):
    # scatter_plot.save('age_vs_hr_scatter.html')
    # scatter_plot.save('age_vs_hr_scatter.json') 
    # scatter_plot.save('age_vs_hr_scatter.png') # Requires vl-convert
    ```
    ![Altair scatter plot python gender age heart rate example output chart](media/altair_scatter_plot_python_gen_05.png)
    <!---
    *   This example creates a scatter plot showing the relationship between age and heart rate, with points colored by gender.
    *   Tooltips allow users to see specific data values when they hover over a point.
    *   The `.interactive()` call enables basic zoom and pan functionality.
    --->

### 2.4. Building Blocks for Dynamic Charts (e.g., for Gapminder-style visualization)
To create more advanced interactive charts, like the Gapminder-style visualization we'll aim for in the Streamlit demo, we need a few more Altair concepts. This section focuses on the Altair techniques for creating components that can be assembled into such visualizations.

*   **Selections:** Selections are the core of Altair's interactivity. They define how users can interact with the chart.
    *   `alt.selection_interval()`: Allows selecting a rectangular region (brushing).
    *   `alt.selection_point()`: Allows selecting single or multiple discrete points.
    *   `alt.selection_single()`: Allows selecting a single discrete item, often used with `bind` for widgets.
*   **Input Binding (for `selection_single`):** Connects a selection to an HTML input element.
    *   `bind=alt.binding_range(min=V, max=V, step=V)`: Creates a slider.
    *   `bind=alt.binding_select(options=[...])`: Creates a dropdown menu.
*   **Conditional Encodings:** Change visual properties based on a selection.
    *   `alt.condition(selection, value_if_selected, value_if_not_selected)`
    *   Example: `color=alt.condition(my_selection, 'steelblue', 'lightgray')`
*   **Transformations:** Modify the data before encoding.
    *   `transform_filter(selection_or_expression)`: Filter data based on a selection or a Vega expression.
    *   `transform_aggregate(...)`: Perform aggregations (e.g., mean, sum).
    *   `transform_window(...)`: For window functions (e.g., rank, cumulative sum).
*   **Layering & Concatenation:** Combine multiple chart specifications.
    *   `chart1 + chart2`: Layer charts on top of each other (share axes).
    *   `chart1 | chart2`: Place charts side-by-side (horizontal concatenation).
    *   `chart1 & chart2`: Place charts one above the other (vertical concatenation).

*   **Key Altair features for a Gapminder-style dynamic chart:**
    *   **Data:** A DataFrame with columns for an X-variable (e.g., health expenditure, often log-scaled), a Y-variable (e.g., life expectancy), a size variable (e.g., population), a color variable (e.g., region/country group), and a time variable (e.g., year).
    *   **Time Slider:** Use `alt.selection_single` with `bind=alt.binding_range` to create a slider for the `year` field.
    *   **Filtering:** Use `transform_filter(year_slider_selection)` to filter the data displayed in the chart based on the year selected by the slider.
    *   **Encodings:** Map the data columns to `x`, `y`, `size`, and `color` visual channels.
    *   **Tooltips:** Provide rich information on hover.
    *   **Scales:** May need to customize scales (e.g., `alt.Scale(type="log")` for the x-axis).

```python
# Conceptual Altair components for a dynamic time-series scatter plot:
# This code illustrates the Altair setup for a chart that could be part of a Gapminder-style dashboard.
# The full interactive dashboard experience will be built using Streamlit later.

# import altair as alt
# import pandas as pd

# # Assume health_data_df exists. For example:
# # health_data_df = pd.DataFrame({
# # 'country': ['A', 'B', 'A', 'B'], 'year': [2000, 2000, 2001, 2001],
# # 'health_exp_pc': [100, 200, 110, 220], 'life_expectancy': [70, 72, 71, 73],
# # 'population': [1e6, 2e6, 1.1e6, 2.1e6], 'region': ['R1', 'R2', 'R1', 'R2']
# # })

# # Create a selection for the year slider
# year_slider = alt.selection_single(
#     name="Select_Year",  # Name for the selection
#     fields=['year'],     # Data field to bind to
#     # bind=alt.binding_range(min=health_data_df['year'].min(), max=health_data_df['year'].max(), step=1), # Create slider
#     # init={'year': health_data_df['year'].min()} # Initial year
# )

# # Base chart definition
# gapminder_like_chart_spec = alt.Chart(health_data_df).mark_circle(opacity=0.7).encode(
#     x=alt.X('health_exp_pc:Q', scale=alt.Scale(type="log"), title='Health Expenditure per Capita (log scale)'),
#     y=alt.Y('life_expectancy:Q', scale=alt.Scale(zero=False), title='Life Expectancy'),
#     size=alt.Size('population:Q', title='Population', scale=alt.Scale(range=[50, 2000])), # Adjust size range for visibility
#     color=alt.Color('region:N', title='Region'),
#     tooltip=[
#         alt.Tooltip('country:N', title='Country'),
#         alt.Tooltip('year:O', title='Year'),
#         alt.Tooltip('life_expectancy:Q', title='Life Expectancy', format='.1f'),
#         alt.Tooltip('health_exp_pc:Q', title='Health Exp/Capita', format='$,.0f'),
#         alt.Tooltip('population:Q', title='Population', format=',.0f')
#     ]
# ).add_params(
#     year_slider  # Add the slider selection to the chart
# ).transform_filter(
#     year_slider  # Filter the data based on the slider
# ).properties(
#     width=650,
#     height=400,
#     title="Health & Wealth Over Time (Conceptual Components)"
# )

# # To save this specification for later use (e.g., in Streamlit or MkDocs):
# # gapminder_like_chart_spec.save('gapminder_altair_spec.json')
```
![Gapminder style altair chart life expectancy health expenditure population single year concept visualization](#FIXME)
<!---
*   This conceptual code outlines how to define an Altair chart with a time slider.
*   The `selection_single` with `binding_range` creates the slider.
*   `transform_filter` dynamically updates the chart based on the slider's current year.
*   The actual data loading and precise binding would be part of the demo implementation.
*   This JSON specification can then be used by Streamlit to render the interactive chart.
--->

### Demo 2: Interactive Altair Chart
*   (Refer to [`lectures/09/demo/02_altair_interactive_chart.md`](lectures/09/demo/02_altair_interactive_chart.md))
    <!---
    *   This demo will involve creating a simpler interactive chart, perhaps with a categorical filter or a brush selection, using a real PhysioNet dataset.
    *   It reinforces the concepts of selections and saving for embedding.
    --->

---

## 3. Automated Report Generation with MkDocs (20 minutes)

Once you have created insightful visualizations and diagrams, you need an effective way to share them along with your narrative and findings. MkDocs is a static site generator that allows you to create professional-looking project documentation and reports using Markdown.

### 3.1. Why Static Site Generators for Reports?
*   **Concept & Benefits:** Static site generators (SSGs) like MkDocs take source files (e.g., Markdown text, images, chart specifications) and templates to produce a complete, self-contained HTML website. For data science reports, this offers:
    *   **Shareability:** Simple HTML files are easy to host on a web server, GitHub Pages, or send as a zipped archive.
    *   **Version Control:** The entire report source (Markdown, Python scripts for generating charts, configuration files) can be managed with Git.
    *   **Reproducibility:** Reports can be consistently rebuilt from the source files at any time.
    *   **Professional Appearance:** Themes (like Material for MkDocs) provide a polished look with minimal effort.
    *   **Automation:** The process of generating charts and building the report can be scripted.
    <!---
    *   SSGs bridge the gap between writing analysis code and producing a presentable, shareable output.
    *   They are an excellent alternative to manually assembling reports in word processors or relying solely on Jupyter Notebooks for dissemination.
    --->
*   **MkDocs:** MkDocs is known for its speed, simplicity, and focus on creating project documentation, which extends well to generating data analysis reports. It uses Markdown for content, making it easy to write.
    ![MkDocs official logo](media/mkdocs_official_logo_03.png)
    ![MkDocs material official logo](media/mkdocs_material_official_logo_04.png)

### 3.2. Setting up MkDocs
*   **Installation:** You'll need MkDocs itself, a theme (Material for MkDocs is highly recommended), and any plugins. For embedding Altair charts, we'll use `mkdocs-altair-plugin`.
    ```bash
    pip install mkdocs mkdocs-material mkdocs-altair-plugin pandas altair
    ```
    <!---
    *   `mkdocs`: The core static site generator.
    *   `mkdocs-material`: A popular and feature-rich theme for MkDocs.
    *   `mkdocs-altair-plugin`: Allows easy embedding of Altair charts.
    *   `pandas` and `altair`: Needed if your report generation process involves creating charts with Python.
    --->
*   **Project Initialization:** To start a new MkDocs project:
    ```bash
    mkdocs new my_health_report
    cd my_health_report
    ```
    This creates a basic project structure:
    ```
    my_health_report/
    ├── mkdocs.yml    # The main configuration file
    └── docs/
        └── index.md  # The homepage for your report
    ```
    <!---
    *   The `mkdocs new` command sets up the essential files and directories.
    *   You'll primarily work within the `docs/` directory for content and edit `mkdocs.yml` for configuration.
    --->
*   **Directory Structure (Recommended):** It's good practice to organize supporting files. For example, create a `docs/charts/` directory for Altair JSON specifications and `docs/media/` for images.
    ```
    my_health_report/
    ├── mkdocs.yml
    └── docs/
        ├── index.md
        ├── analysis_page.md
        ├── charts/
        │   └── my_altair_chart.json
        └── media/
            └── workflow_diagram.png 
    ```

### 3.3. Configuring `mkdocs.yml`
The `mkdocs.yml` file controls your site's settings, theme, navigation, and plugins.

*   **Basic Configuration:**
    ```yaml
    site_name: My Health Data Report
    site_description: 'A report on health data analysis findings.'
    site_author: 'Your Name'

    theme:
      name: material  # Using the Material for MkDocs theme
      # Optional: add features, palette, logo, etc.
      # features:
      #   - navigation.tabs
      # palette:
      #   primary: 'indigo'
      #   accent: 'blue'
      # logo: media/logo.png 
    ```
    ![MkDocs material theme example website screenshot documentation site](media/mkdocs_material_theme_example__08.png)
*   **Plugins:** Enable plugins, especially for Altair charts.
    ```yaml
    plugins:
      - search        # Built-in search plugin
      - altair:       # mkdocs-altair-plugin configuration
          vega_lite_version: "5" # Specify Vega-Lite version consistent with your Altair
          # embed_options: # Optional: customize how charts are embedded
          #   actions: false # Example: hide the actions (save, view source) menu on charts
    ```
    <!---
    *   The `mkdocs-material` theme offers many customization options documented on its website.
    *   Ensure the `vega_lite_version` in the `altair` plugin matches the version Altair is using to avoid rendering issues.
    --->
*   **Navigation (`nav`):** Defines the structure of your site's navigation menu.
    ```yaml
    nav:
      - 'Home': 'index.md'
      - 'Analysis Details':
        - 'Part 1: EDA': 'eda.md'
        - 'Part 2: Modeling': 'modeling.md'
      - 'Interactive Charts': 'interactive_charts.md'
      - 'About': 'about.md'
    ```
    <!---
    *   The `nav` section allows you to create a hierarchical menu for your report pages.
    --->

### 3.4. Creating Report Content & Embedding Charts/Diagrams
*   **Markdown:** Write your report narrative, analysis, and findings in `.md` files within the `docs/` directory. Standard Markdown syntax applies.
*   **Python Script for Charts:** It's good practice to have a separate Python script (e.g., in a `scripts/` directory at the project root, or directly in `docs/` if simple) that generates your Altair charts and saves them as JSON files into a designated folder, like `docs/charts/`.
    ```python
    # Example: scripts/generate_report_charts.py
    # import altair as alt
    # import pandas as pd
    # from pathlib import Path

    # # Assume physio_df is loaded or created
    # # ... (chart creation code from section 2.3) ...
    # # scatter_plot = alt.Chart(physio_df).mark_point()... 

    # output_dir = Path("../docs/charts") # Relative to script location if script is in scripts/
    # output_dir.mkdir(parents=True, exist_ok=True)
    # # scatter_plot.save(output_dir / "age_vs_hr_scatter.json")
    # print(f"Saved chart to {output_dir / 'age_vs_hr_scatter.json'}")
    ```
    <!---
    *   This script would be run manually or as part of an automated build process *before* running `mkdocs build`.
    --->
*   **Embedding Altair Charts:** In your Markdown files, use the tag provided by `mkdocs-altair-plugin`:
    ```markdown
    Here is an interactive chart showing age vs. heart rate:

    {% include_altair "charts/age_vs_hr_scatter.json" %}

    The chart shows a positive correlation...
    ```
    <!---
    *   The path to the JSON file is relative to the `docs/` directory.
    --->
*   **Embedding Mermaid Diagrams:** If your MkDocs theme (like Material for MkDocs) supports it, you can embed Mermaid diagrams directly in your Markdown using standard Mermaid fenced code blocks:
    ```markdown
    This workflow was followed:

    ```mermaid
    graph TD;
        A[Data Collection] --> B(Processing);
        B --> C[Analysis];
        C --> D[Report Generation];
    ```
    <!---
    *   Material for MkDocs includes support for Mermaid out-of-the-box. For other themes, a plugin like `mkdocs-mermaid2-plugin` might be needed.
    --->

### 3.5. Building, Serving, and Deploying
*   **Build:** To generate the static HTML site:
    ```bash
    mkdocs build
    ```
    This creates a `site/` directory containing all the HTML, CSS, and JS files for your report.
*   **Serve Locally:** To preview your report locally with live reloading as you make changes:
    ```bash
    mkdocs serve
    ```
    This usually starts a server at `http://127.0.0.1:8000`.
*   **Deploying with GitHub Pages (Conceptual Overview):**
    GitHub Pages is a free way to host your static MkDocs site directly from a GitHub repository.
    1.  Ensure your MkDocs project is a GitHub repository.
    2.  Install `ghp-deploy`: `pip install ghp-deploy`.
    3.  Run: `mkdocs gh-deploy`. This command builds your site and pushes the `site/` contents to a special `gh-pages` branch on GitHub, which then serves the site.
    4.  Alternatively, GitHub Actions can be configured to automate this deployment on every push to your main branch.
    <!---
    *   The `site/` directory is what gets deployed. It's entirely self-contained.
    *   `mkdocs gh-deploy` simplifies the deployment process to GitHub Pages significantly.
    --->

### Demo 3: Automated Report with MkDocs
*   (Refer to [`lectures/09/demo/03_mkdocs_automated_report.md`](lectures/09/demo/03_mkdocs_automated_report.md))
*   This demo will include generating an Altair chart, saving it as JSON, embedding it in an MkDocs page, and also embedding a Mermaid diagram (e.g., a workflow for the data used in the report).
    <!---
    *   The demo will walk through setting up a minimal MkDocs site, creating content, embedding both an Altair chart and a Mermaid diagram, and serving it locally.
    --->

---

## 4. Interactive Dashboards with Streamlit (25 minutes)

While MkDocs is excellent for creating static reports with embedded interactive charts, sometimes you need a more dynamic application where users can manipulate inputs, trigger computations, and see results update live. Streamlit is a Python library designed for rapidly building and sharing such data applications.

### 4.1. From Reports to Interactive Applications
*   **What is Streamlit?** Streamlit is an open-source Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data science. It's often described as "the fastest way to build and share data apps."
    <!---
    *   Streamlit allows you to build interactive UIs directly from your Python scripts with minimal overhead.
    --->
*   **Why Streamlit?**
    *   **Rapid Prototyping:** Go from Python script to interactive web app in minutes.
    *   **Python-centric:** Write apps using only Python; no HTML, CSS, or JavaScript knowledge is required for basic apps.
    *   **Interactive Widgets:** Comes with a rich set of input widgets (sliders, dropdowns, text inputs, etc.) that are easy to implement.
    *   **Easy to Share:** Streamlit Community Cloud offers free deployment for public apps.
    <!---
    *   Streamlit is particularly useful when you want to provide a tool for others (even non-programmers) to explore data or model predictions by changing parameters.
    *   It bridges the gap between a data analysis script and a user-friendly web application.
    --->
    ![Streamlit official logo](media/streamlit_official_logo_05.png)

### 4.2. Streamlit Fundamentals
*   **Installation:**
    ```bash
    pip install streamlit altair pandas
    ```
    <!---
    *   `streamlit`: The core library.
    *   `altair`, `pandas`: Often used with Streamlit to create and display data/visualizations.
    --->
*   **Running an App:**
    Save your Streamlit code as a Python file (e.g., `my_app.py`) and run it from your terminal:
    ```bash
    streamlit run my_app.py
    ```
    This will typically open the app in your default web browser.
*   **Core Concepts:**
    *   **Script Reruns:** Streamlit apps rerun your Python script from top to bottom whenever a user interacts with a widget or the app needs to update. This is a fundamental concept to grasp.
    *   **Layout Commands:**
        *   `st.title("My App Title")`
        *   `st.header("Section Header")`
        *   `st.subheader("Sub-Section")`
        *   `st.write("Some text or a Python variable.")`
        *   `st.markdown("Supports **Markdown** formatting.")`
        *   `st.sidebar`: Used to place elements in a sidebar (e.g., `st.sidebar.header("Filters")`).
    *   **Displaying Data:**
        *   `st.dataframe(my_pandas_df)`: Displays a Pandas DataFrame as an interactive table.
        *   `st.table(my_pandas_df)`: Displays a static table.
        *   `st.metric(label="Metric Name", value=123, delta="-5%")`: Displays a single metric with an optional change indicator.
        *   `st.json(my_dict_or_list)`: Displays JSON.
    *   **Displaying Charts:**
        *   `st.altair_chart(my_altair_chart_object, use_container_width=True)`
        *   `st.pyplot(my_matplotlib_fig_object)`
        *   `st.plotly_chart(my_plotly_fig_object)`
    *   **Input Widgets:** These are functions that return the current value selected by the user.
        *   `st.button("Click me")`: Returns `True` when clicked.
        *   `selected_value = st.slider("Select a range", 0, 100, (25, 75))`: Returns a tuple for a range slider.
        *   `option = st.selectbox("Choose an option", ('A', 'B', 'C'))`: Returns the selected option.
        *   `options = st.multiselect("Choose multiple", ['X', 'Y', 'Z'])`: Returns a list of selected options.
        *   `text = st.text_input("Enter text", "Default value")`
        *   `date = st.date_input("Pick a date")`
    *   **Caching:** Use `@st.cache_data` or `@st.cache_resource` decorators to cache the results of expensive functions (like data loading or model computation) to improve performance, as the script reruns frequently.
    <!---
    *   The simplicity of Streamlit's API allows for quick iteration.
    *   Understanding the rerun behavior is key to managing state and performance in more complex apps.
    *   Widgets are the primary way users interact with and control the app.
    --->
    ![Streamlit app example interface common widgets screenshot python dashboard](media/streamlit_app_example_interfac_10.png)

### 4.3. Demo 4: Simple, Practical Health Dashboard with Streamlit
*   (Refer to [`lectures/09/demo/04_streamlit_simple_dashboard.md`](lectures/09/demo/04_streamlit_simple_dashboard.md))
*   **Goal:** Demonstrate immediate utility with a common task using a PhysioNet dataset snippet.
*   **Key features to showcase:**
    *   Loading data (using `@st.cache_data` for efficiency).
    *   Using `st.sidebar` for filters (e.g., `st.selectbox`, `st.slider`).
    *   Filtering a Pandas DataFrame based on widget selections.
    *   Displaying the filtered DataFrame (`st.dataframe`).
    *   Displaying a simple Altair chart based on the filtered data (`st.altair_chart`).
    *   Using `st.metric` to show a key summary statistic.
    <!---
    *   This demo focuses on the fundamental workflow: input widgets -> data manipulation -> output display.
    *   It highlights how quickly a useful data exploration tool can be built.
    --->

### 4.4. Demo 5: Mini-Gapminder Style Dashboard with Streamlit
*   (Refer to [`lectures/09/demo/05_streamlit_gapminder_dashboard.md`](lectures/09/demo/05_streamlit_gapminder_dashboard.md))
*   **Goal:** Showcase more dynamic interactivity and data storytelling, building on Altair components.
*   **Key features to showcase:**
    *   Using an appropriate dataset (public Gapminder or a suitable health equivalent).
    *   Implementing a `st.slider` for "Year" selection.
    *   Dynamically filtering the DataFrame passed to an Altair chart based on the slider.
    *   The Altair chart itself will be the multi-encoded scatter plot (X, Y, Size, Color) discussed in section 2.4.
    *   Displaying the interactive Altair chart using `st.altair_chart`.
    <!---
    *   This demo illustrates how Streamlit can host more complex, interactive Altair visualizations.
    *   The Streamlit slider controls the year, and the Altair chart updates dynamically based on the filtered data for that year.
    --->

### 4.5. Sharing Streamlit Apps (Briefly)
*   **Streamlit Community Cloud:** Streamlit offers a free platform called Streamlit Community Cloud for deploying and sharing public Streamlit apps directly from GitHub repositories.
    <!---
    *   This is the easiest way for students to share their projects.
    *   Requires pushing the app script and a `requirements.txt` file to GitHub.
    --->
*   **Other Deployment Options (Conceptual Mention):** For private apps or more complex needs, Streamlit apps can also be deployed using Docker containers on various cloud platforms (AWS, GCP, Azure) or on-premise servers.
    <!---
    *   These options offer more control but involve more setup.
    --->

---