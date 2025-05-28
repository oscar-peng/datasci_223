# Lecture 09 Demos

This directory contains demo materials for Lecture 09 on Data Visualization and Dashboarding.

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Demos

### 1. Altair Chart Generation

To generate the example Altair charts:

```bash
python generate_charts.py
```

This will create JSON specifications for various charts in the `assets/charts` directory.

### 2. Simple Dash Dashboard

To run the interactive Dash dashboard:

```bash
python simple_dash_dashboard.py
```

Then open your browser and navigate to `http://127.0.0.1:8050/`

The dashboard includes:
- A dropdown to filter by patient condition
- An age distribution histogram
- A scatter plot of blood pressure vs heart rate
- Interactive features like zooming and hovering

## Expected Outputs

### Altair Charts
- Basic scatter plot with tooltips
- Scatter plot with marginal histograms
- Interactive variable selection chart

### Dash Dashboard
- Interactive web interface
- Real-time chart updates based on filter selection
- Responsive layout that works on different screen sizes

## Troubleshooting

If you encounter any issues:

1. Ensure all dependencies are installed correctly
2. Check that you're using Python 3.8 or higher
3. Verify that the virtual environment is activated
4. Make sure no other process is using port 8050 (for the Dash app)

For additional help, please refer to the lecture materials or contact the instructor. 