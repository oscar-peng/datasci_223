# Automated Clinical Research Reports with MkDocs

This project demonstrates how to create automated, interactive clinical research reports and publish them to GitHub Pages. It's designed for health data science students who are new to programming and want to learn modern documentation practices.

## 🎯 Learning Objectives

By the end of this demo, you will be able to:
- Generate automated statistical reports from clinical data
- Create interactive data visualizations
- Build professional documentation sites
- Deploy reports to GitHub Pages
- Understand basic version control practices

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.8 or higher
- Git installed on your computer
- A GitHub account

### 2. Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/neonatal-feeding-study.git
cd neonatal-feeding-study
```

2. **Create and activate a virtual environment:**
```bash
# On macOS/Linux:
python -m venv venv
source venv/bin/activate

# On Windows:
python -m venv venv
venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## 📊 Running the Pipeline

### 1. Generate Reports

The pipeline will:
- Generate sample neonatal feeding study data
- Perform statistical analysis
- Create interactive visualizations
- Build the documentation site

```bash
python orchestrator.py
```

### 2. View Your Report Locally

```bash
mkdocs serve
```
Then open your browser to: http://127.0.0.1:8000

### 3. Deploy to GitHub Pages

1. Create a new GitHub repository
2. Update `mkdocs.yml` with your repository details:
   ```yaml
   site_url: 'https://yourusername.github.io/neonatal-feeding-study'
   repo_name: 'yourusername/neonatal-feeding-study'
   repo_url: 'https://github.com/yourusername/neonatal-feeding-study'
   ```
3. Push your changes:
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```
4. Enable GitHub Pages in your repository settings

## 🎓 Learning Resources

### Key Concepts

1. **MkDocs**
   - A static site generator for project documentation
   - Uses Markdown for content
   - Supports themes and plugins

2. **Altair/Vega-Lite**
   - Declarative visualization library
   - Creates interactive charts
   - JSON-based specification

3. **GitHub Pages**
   - Free hosting for static websites
   - Automatic deployment via GitHub Actions
   - Custom domain support

### Common Tasks

1. **Adding a New Chart**
   ```python
   import altair as alt
   
   chart = alt.Chart(data).mark_point().encode(
       x='gestational_age',
       y='time_to_fof',
       color='ventilation_status'
   )
   ```

2. **Creating a New Report Section**
   ```markdown
   # New Analysis Section
   
   ## Methods
   Describe your statistical methods here...
   
   ## Results
   Present your findings here...
   
   ```vegalite
   {
     "data": {...},
     "mark": "point",
     "encoding": {...}
   }
   ```
   ```

## 🔍 Troubleshooting

### Common Issues

1. **MkDocs Build Fails**
   - Check that all dependencies are installed
   - Verify your `mkdocs.yml` configuration
   - Ensure all chart JSON files are valid

2. **Charts Not Displaying**
   - Check browser console for errors
   - Verify Vega-Lite JSON syntax
   - Ensure data files are in correct location

3. **GitHub Pages Not Updating**
   - Check GitHub Actions workflow status
   - Verify repository settings
   - Ensure `mkdocs.yml` has correct site URL

## 📚 Additional Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Altair Documentation](https://altair-viz.github.io/)
- [GitHub Pages Guide](https://pages.github.com/)
- [Markdown Guide](https://www.markdownguide.org/)

## 🤝 Contributing

Feel free to:
- Report issues
- Suggest improvements
- Submit pull requests

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details. 