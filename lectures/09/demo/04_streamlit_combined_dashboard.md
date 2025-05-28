# Demo 4: Building Interactive Health Dashboards with Streamlit

**Overall Goal:** Learn to build both simple, practical health data dashboards and more complex, dynamic Gapminder-style visualizations using Streamlit and Altair.

**Part 1: Simple Clinical Data Explorer**
**Part 2: Advanced Gapminder-Style Health & Wealth Dashboard**

## Prerequisites

Ensure you have a directory for this demo and the necessary packages installed.
Create a new directory (e.g., `streamlit_health_dashboards`) and navigate into it.

```bash
mkdir streamlit_health_dashboards
cd streamlit_health_dashboards
pip install streamlit altair pandas numpy
```

## Part 1: Simple Clinical Data Explorer

**Goal:** Build a basic Streamlit app to load, filter, and visualize a snippet of clinical data.

### Step 1.1: Create the App File and Load Data

Create a Python file named `simple_clinical_app.py`.

```python
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# --- Data Generation (Simplified for this part) ---
@st.cache_data # Cache the data to prevent reloading on every interaction
def load_simple_data():
    np.random.seed(42)
    num_patients = 100
    data = pd.DataFrame({
        'patient_id': [f'PAT{i:03}' for i in range(num_patients)],
        'age': np.random.randint(20, 80, num_patients),
        'gender': np.random.choice(['Male', 'Female'], num_patients, p=[0.45, 0.55]),
        'condition': np.random.choice(['Healthy', 'Type 2 Diabetes', 'Hypertension', 'Asthma'], num_patients, p=[0.4, 0.2, 0.25, 0.15]),
        'systolic_bp': np.random.randint(100, 180, num_patients),
        'cholesterol': np.random.randint(150, 300, num_patients)
    })
    # Adjust BP based on condition
    data.loc[data['condition'] == 'Hypertension', 'systolic_bp'] += np.random.randint(10, 30, size=len(data[data['condition'] == 'Hypertension']))
    data['systolic_bp'] = data['systolic_bp'].clip(upper=220)
    return data

df_simple = load_simple_data()

# --- Streamlit App Layout ---
st.set_page_config(page_title="Simple Clinical Explorer", layout="wide")

st.title("🩺 Simple Clinical Data Explorer")
st.markdown("Use the sidebar to filter and explore patient data.")

# --- Sidebar for Filters ---
st.sidebar.header("📊 Filters")

# Age range slider
age_range = st.sidebar.slider(
    "Select Age Range:",
    min_value=int(df_simple['age'].min()),
    max_value=int(df_simple['age'].max()),
    value=(int(df_simple['age'].min()), int(df_simple['age'].max()))
)

# Gender selectbox
gender_options = ['All'] + list(df_simple['gender'].unique())
selected_gender = st.sidebar.selectbox(
    "Select Gender:",
    options=gender_options,
    index=0
)

# Condition multiselect
condition_options = ['All'] + list(df_simple['condition'].unique())
selected_conditions = st.sidebar.multiselect(
    "Select Medical Conditions:",
    options=df_simple['condition'].unique(), # Provide actual options without 'All'
    default=list(df_simple['condition'].unique())[0:2] # Default to first two for demo
)


# --- Filtering Data ---
df_filtered = df_simple[
    (df_simple['age'] >= age_range[0]) & (df_simple['age'] <= age_range[1])
]

if selected_gender != 'All':
    df_filtered = df_filtered[df_filtered['gender'] == selected_gender]

if selected_conditions: # Check if list is not empty
    if 'All' not in selected_conditions: # Only filter if 'All' is not selected among others
        df_filtered = df_filtered[df_filtered['condition'].isin(selected_conditions)]
else: # If selected_conditions is empty (e.g. user deselects all)
    st.warning("No conditions selected. Displaying data for all conditions.")
    # Optionally, display all data or an empty set based on desired behavior
    # df_filtered = pd.DataFrame(columns=df_simple.columns) # Example: display empty

# --- Main Page Display ---

# Key Metrics
st.subheader("📈 Key Metrics for Filtered Data")
col1, col2, col3 = st.columns(3)
col1.metric("Total Patients", f"{df_filtered.shape[0]}")
col2.metric("Avg. Systolic BP", f"{df_filtered['systolic_bp'].mean():.1f} mmHg" if not df_filtered.empty else "N/A")
col3.metric("Avg. Cholesterol", f"{df_filtered['cholesterol'].mean():.1f} mg/dL" if not df_filtered.empty else "N/A")

st.markdown("---")

# Display Filtered Data Table
st.subheader("📋 Filtered Patient Data")
if not df_filtered.empty:
    st.dataframe(df_filtered, height=300)
else:
    st.info("No data matches the current filter criteria.")

st.markdown("---")

# Simple Altair Chart
st.subheader("📊 Age vs. Systolic BP (Filtered)")
if not df_filtered.empty:
    scatter_chart = alt.Chart(df_filtered).mark_circle(size=100, opacity=0.7).encode(
        x=alt.X('age:Q', title='Age (Years)'),
        y=alt.Y('systolic_bp:Q', title='Systolic BP (mmHg)'),
        color=alt.Color('condition:N', title='Condition'),
        tooltip=['patient_id', 'age', 'gender', 'condition', 'systolic_bp', 'cholesterol']
    ).properties(
        title='Age vs. Systolic Blood Pressure by Condition'
    ).interactive()
    st.altair_chart(scatter_chart, use_container_width=True)
else:
    st.info("Cannot display chart as no data matches the current filter criteria.")

st.sidebar.markdown("---")
st.sidebar.info("This is Part 1 of the Streamlit demo. Part 2 will build a more advanced Gapminder-style dashboard.")

```

### Step 1.2: Run the Simple App

Save the file and run it from your terminal:
```bash
streamlit run simple_clinical_app.py
```
Interact with the filters in the sidebar and observe how the metrics, data table, and chart update.

**Key features demonstrated in Part 1:**
-   [`@st.cache_data`](https://docs.streamlit.io/library/api-reference/performance/st.cache_data) for efficient data loading.
-   [`st.sidebar`](https://docs.streamlit.io/library/api-reference/layout/st.sidebar) for organizing input widgets.
-   Widgets: [`st.slider`](https://docs.streamlit.io/library/api-reference/widgets/st.slider), [`st.selectbox`](https://docs.streamlit.io/library/api-reference/widgets/st.selectbox), [`st.multiselect`](https://docs.streamlit.io/library/api-reference/widgets/st.multiselect).
-   Dynamic filtering of a Pandas DataFrame.
-   Displaying data with [`st.metric`](https://docs.streamlit.io/library/api-reference/data/st.metric) and [`st.dataframe`](https://docs.streamlit.io/library/api-reference/data/st.dataframe).
-   Embedding an interactive Altair chart with [`st.altair_chart`](https://docs.streamlit.io/library/api-reference/charts/st.altair_chart).

---

## Part 2: Advanced Gapminder-Style Health & Wealth Dashboard

**Goal:** Create a dynamic, Gapminder-inspired dashboard showing health and economic indicators over time, with animated transitions and interactive controls. This part builds upon the concepts from Part 1 and incorporates more advanced features.

*(Content from `lectures/09/demo/05_streamlit_gapminder_dashboard.md` will be integrated here, starting with its data generation script and then its Streamlit app code, possibly in a new file or by extending `simple_clinical_app.py`)*


### Step 2.1: Create Health & Economic Dataset (Gapminder Style)

This step uses the data generation script from the original `05_streamlit_gapminder_dashboard.md`. Create a Python file named `generate_gapminder_data.py` in the same `streamlit_health_dashboards` directory with the following content:

```python
import pandas as pd
import numpy as np
import json

def generate_gapminder_health_data():
    """Generate Gapminder-style health and economic data."""
    np.random.seed(42)
    
    # Define countries and regions
    countries_regions = {
        'North America': ['United States', 'Canada', 'Mexico'],
        'Europe': ['Germany', 'France', 'United Kingdom', 'Italy', 'Spain', 'Netherlands'],
        'Asia': ['China', 'India', 'Japan', 'South Korea', 'Thailand', 'Indonesia'],
        'Africa': ['Nigeria', 'South Africa', 'Kenya', 'Egypt', 'Ghana', 'Morocco'],
        'South America': ['Brazil', 'Argentina', 'Chile', 'Colombia', 'Peru'],
        'Oceania': ['Australia', 'New Zealand']
    }
    
    # Flatten countries list
    all_countries = []
    country_to_region = {}
    for region, countries in countries_regions.items():
        all_countries.extend(countries)
        for country in countries:
            country_to_region[country] = region
    
    # Generate data for years 2000-2023
    years = list(range(2000, 2024))
    data = []
    
    for country in all_countries:
        region = country_to_region[country]
        
        # Set baseline values based on region (realistic starting points)
        if region == 'North America':
            base_life_exp = 77 + np.random.normal(0, 2)
            base_health_exp = 8000 + np.random.normal(0, 1000)
            base_population = 50_000_000 + np.random.normal(0, 20_000_000)
        elif region == 'Europe':
            base_life_exp = 78 + np.random.normal(0, 2)
            base_health_exp = 4000 + np.random.normal(0, 800)
            base_population = 30_000_000 + np.random.normal(0, 15_000_000)
        elif region == 'Asia':
            base_life_exp = 70 + np.random.normal(0, 5)
            base_health_exp = 1500 + np.random.normal(0, 500)
            base_population = 100_000_000 + np.random.normal(0, 200_000_000)
        elif region == 'Africa':
            base_life_exp = 60 + np.random.normal(0, 5)
            base_health_exp = 500 + np.random.normal(0, 200)
            base_population = 40_000_000 + np.random.normal(0, 30_000_000)
        elif region == 'South America':
            base_life_exp = 72 + np.random.normal(0, 3)
            base_health_exp = 1200 + np.random.normal(0, 300)
            base_population = 25_000_000 + np.random.normal(0, 15_000_000)
        else:  # Oceania
            base_life_exp = 80 + np.random.normal(0, 1)
            base_health_exp = 5000 + np.random.normal(0, 500)
            base_population = 15_000_000 + np.random.normal(0, 10_000_000)
        
        # Ensure positive population
        base_population = max(1_000_000, base_population)
        
        for year in years:
            # Calculate year progression (0 to 1 over the time period)
            year_progress = (year - 2000) / (2023 - 2000)
            
            # Life expectancy generally increases over time
            life_expectancy = base_life_exp + year_progress * np.random.uniform(2, 8)
            life_expectancy += np.random.normal(0, 0.5)  # Annual variation
            
            # Health expenditure generally increases (with some economic cycles)
            health_exp_per_capita = base_health_exp * (1 + year_progress * np.random.uniform(0.3, 1.2))
            # Add economic cycles
            cycle_effect = np.sin((year - 2000) * 0.5) * 0.1
            health_exp_per_capita *= (1 + cycle_effect)
            health_exp_per_capita += np.random.normal(0, health_exp_per_capita * 0.05)
            
            # Population grows over time
            population = base_population * (1 + year_progress * np.random.uniform(0.1, 0.4))
            population += np.random.normal(0, population * 0.02)
            
            # Infant mortality rate (decreases over time, inversely related to health spending)
            base_infant_mortality = max(2, 100 - base_life_exp * 1.2)
            infant_mortality = base_infant_mortality * (1 - year_progress * 0.4)
            infant_mortality += np.random.normal(0, 1)
            infant_mortality = max(1, infant_mortality)
            
            # Healthcare access score (0-100, improves over time)
            healthcare_access = min(100, (base_health_exp / 100) + year_progress * 20)
            healthcare_access += np.random.normal(0, 3)
            healthcare_access = max(10, min(100, healthcare_access))
            
            # Add some special events (pandemics, economic crises)
            if year == 2008:  # Financial crisis
                health_exp_per_capita *= 0.95
                life_expectancy -= 0.1
            elif year in [2020, 2021]:  # COVID-19 pandemic
                life_expectancy -= np.random.uniform(0.5, 2.0)
                health_exp_per_capita *= np.random.uniform(1.05, 1.15)  # Increased health spending
            
            data.append({
                'country': country,
                'region': region,
                'year': year,
                'life_expectancy': round(max(40, life_expectancy), 1),
                'health_expenditure_per_capita': round(max(50, health_exp_per_capita), 0),
                'population': int(max(500_000, population)),
                'infant_mortality_rate': round(max(1, infant_mortality), 1),
                'healthcare_access_score': round(max(10, min(100, healthcare_access)), 1)
            })
    
    return pd.DataFrame(data)

def save_sample_data():
    """Generate and save the dataset."""
    df = generate_gapminder_health_data()
    df.to_csv('gapminder_health_data.csv', index=False)
    
    # Save summary statistics
    summary = {
        'countries': len(df['country'].unique()),
        'regions': len(df['region'].unique()),
        'years': len(df['year'].unique()),
        'total_records': len(df),
        'year_range': f"{df['year'].min()}-{df['year'].max()}"
    }
    
    with open('data_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Generated {len(df)} records for {len(df['country'].unique())} countries")
    print(f"📊 Years: {df['year'].min()}-{df['year'].max()}")
    print(f"🌍 Regions: {', '.join(df['region'].unique())}")
    
    return df

if __name__ == "__main__":
    df = save_sample_data()
    print("\nSample data:")
    print(df.head(10))
```

After creating `generate_gapminder_data.py`, run it to generate the `gapminder_health_data.csv` file:
```bash
python generate_gapminder_data.py
```

### Step 2.2: Create the Advanced Gapminder Dashboard App

Now, create a new Python file named `advanced_gapminder_app.py` (or you can extend `simple_clinical_app.py`). For clarity, we'll use a new file here.

```python
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import time

# Configure page
st.set_page_config(
    page_title="Health & Wealth Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .year-display {
        font-size: 4rem;
        font-weight: bold;
        text-align: center;
        color: #667eea;
        margin: 1rem 0;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the Gapminder-style data."""
    try:
        df = pd.read_csv('gapminder_health_data.csv') # Assumes generate_gapminder_data.py has been run
        return df
    except FileNotFoundError:
        st.error("Data file 'gapminder_health_data.csv' not found. Please run generate_gapminder_data.py first.")
        st.stop()

def create_gapminder_chart(df, selected_year, x_var, y_var, size_var, color_var):
    """Create the main Gapminder-style scatter plot."""
    
    # Filter data for selected year
    year_data = df[df['year'] == selected_year].copy()
    
    # Create the chart
    chart = alt.Chart(year_data).mark_circle(
        opacity=0.8,
        stroke='white',
        strokeWidth=1
    ).encode(
        x=alt.X(f'{x_var}:Q',
               title=get_variable_title(x_var),
               scale=alt.Scale(type='log' if x_var == 'health_expenditure_per_capita' else 'linear')),
        y=alt.Y(f'{y_var}:Q',
               title=get_variable_title(y_var),
               scale=alt.Scale(zero=False)),
        size=alt.Size(f'{size_var}:Q',
                     title=get_variable_title(size_var),
                     scale=alt.Scale(range=[100, 1000])),
        color=alt.Color(f'{color_var}:N',
                       title=get_variable_title(color_var),
                       scale=alt.Scale(range=['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33'])),
        tooltip=[
            alt.Tooltip('country:N', title='Country'),
            alt.Tooltip('region:N', title='Region'),
            alt.Tooltip(f'{x_var}:Q', title=get_variable_title(x_var), format='.1f'),
            alt.Tooltip(f'{y_var}:Q', title=get_variable_title(y_var), format='.1f'),
            alt.Tooltip(f'{size_var}:Q', title=get_variable_title(size_var), format=','),
            alt.Tooltip('year:O', title='Year')
        ]
    ).properties(
        width=800,
        height=500,
        title=f"Health & Wealth Indicators - {selected_year}"
    ).interactive()
    
    return chart

def get_variable_title(var_name):
    """Get human-readable titles for variables."""
    titles = {
        'life_expectancy': 'Life Expectancy (years)',
        'health_expenditure_per_capita': 'Health Expenditure per Capita (USD)',
        'population': 'Population',
        'infant_mortality_rate': 'Infant Mortality Rate (per 1000)',
        'healthcare_access_score': 'Healthcare Access Score (0-100)',
        'region': 'Region'
    }
    return titles.get(var_name, var_name.replace('_', ' ').title())

def create_time_series_chart(df, countries, variable):
    """Create a time series chart for selected countries."""
    if not countries:
        return alt.Chart(pd.DataFrame()).mark_line() # Return empty chart if no countries
    
    filtered_df = df[df['country'].isin(countries)]
    
    chart = alt.Chart(filtered_df).mark_line(
        point=True,
        strokeWidth=3
    ).encode(
        x=alt.X('year:O', title='Year'),
        y=alt.Y(f'{variable}:Q', title=get_variable_title(variable)),
        color=alt.Color('country:N', title='Country'),
        tooltip=[
            alt.Tooltip('country:N', title='Country'),
            alt.Tooltip('year:O', title='Year'),
            alt.Tooltip(f'{variable}:Q', title=get_variable_title(variable), format='.1f')
        ]
    ).properties(
        width=600, # Adjusted for potential sidebar layout
        height=300,
        title=f'{get_variable_title(variable)} Over Time'
    )
    
    return chart

def create_regional_summary(df, selected_year):
    """Create regional summary statistics."""
    year_data = df[df['year'] == selected_year]
    
    regional_stats = year_data.groupby('region').agg({
        'life_expectancy': 'mean',
        'health_expenditure_per_capita': 'mean',
        'population': 'sum',
        'infant_mortality_rate': 'mean',
        'healthcare_access_score': 'mean'
    }).round(1)
    
    return regional_stats

def main():
    # Header
    st.markdown('<h1 class="main-header">🌍 Health & Wealth Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Exploring the relationship between health outcomes and economic indicators across countries and time**")
    
    # Load data
    df = load_data()
    
    # Sidebar controls
    st.sidebar.header("🎛️ Dashboard Controls")
    
    # Year selection
    years = sorted(df['year'].unique())
    selected_year = st.sidebar.select_slider(
        "📅 Select Year",
        options=years,
        value=2020, # Default year
        help="Choose a year to explore"
    )
    
    # Display selected year prominently
    # This will be controlled by the animation or the slider
    year_display_placeholder = st.empty()
    year_display_placeholder.markdown(f'<div class="year-display">{selected_year}</div>', unsafe_allow_html=True)
    
    # Variable selection for main chart
    st.sidebar.subheader("📊 Chart Variables")
    
    numeric_vars = ['life_expectancy', 'health_expenditure_per_capita', 'infant_mortality_rate', 'healthcare_access_score']
    
    x_variable = st.sidebar.selectbox(
        "X-axis",
        numeric_vars,
        index=1,  # health_expenditure_per_capita
        help="Variable for horizontal axis"
    )
    
    y_variable = st.sidebar.selectbox(
        "Y-axis",
        numeric_vars,
        index=0,  # life_expectancy
        help="Variable for vertical axis"
    )
    
    size_variable = st.sidebar.selectbox(
        "Bubble Size",
        ['population'] + numeric_vars, # Allow population or other numeric vars for size
        index=0,  # population
        help="Variable determining bubble size"
    )
    
    color_variable = st.sidebar.selectbox(
        "Color",
        ['region'], # Can be expanded to other categorical variables if available
        index=0,
        help="Variable for color coding"
    )
    
    # Animation controls
    st.sidebar.subheader("🎬 Animation")
    
    # Placeholder for the main chart that will be updated during animation
    main_chart_placeholder = st.empty()

    if st.sidebar.button("▶️ Play Animation", help="Animate through years"):
        for year_anim in years:
            year_display_placeholder.markdown(f'<div class="year-display">{year_anim}</div>', unsafe_allow_html=True)
            chart_anim = create_gapminder_chart(df, year_anim, x_variable, y_variable, size_variable, color_variable)
            main_chart_placeholder.altair_chart(chart_anim, use_container_width=True)
            time.sleep(0.5) # Adjust speed of animation
        # Reset to originally selected year after animation
        year_display_placeholder.markdown(f'<div class="year-display">{selected_year}</div>', unsafe_allow_html=True)
        final_chart = create_gapminder_chart(df, selected_year, x_variable, y_variable, size_variable, color_variable)
        main_chart_placeholder.altair_chart(final_chart, use_container_width=True)
    else:
        # Display chart for the initially selected year if not animating
        initial_chart = create_gapminder_chart(df, selected_year, x_variable, y_variable, size_variable, color_variable)
        main_chart_placeholder.altair_chart(initial_chart, use_container_width=True)

    
    # Key insights for the selected year
    year_data_selected = df[df['year'] == selected_year]
    
    st.subheader(f"Global Snapshot for {selected_year}")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_life_exp = year_data_selected['life_expectancy'].mean()
        st.markdown(f"""
        <div class="metric-container">
            <h3>🌟 Avg Life Expectancy</h3>
            <h2>{avg_life_exp:.1f} years</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_pop = year_data_selected['population'].sum()
        st.markdown(f"""
        <div class="metric-container">
            <h3>👥 Total Population</h3>
            <h2>{total_pop/1e9:.1f}B</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_health_exp = year_data_selected['health_expenditure_per_capita'].mean()
        st.markdown(f"""
        <div class="metric-container">
            <h3>💰 Avg Health Spending</h3>
            <h2>${avg_health_exp:,.0f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_infant_mortality = year_data_selected['infant_mortality_rate'].mean()
        st.markdown(f"""
        <div class="metric-container">
            <h3>👶 Avg Infant Mortality</h3>
            <h2>{avg_infant_mortality:.1f}/1000</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Regional breakdown
    st.subheader(f"🌍 Regional Analysis for {selected_year}")
    
    col_reg1, col_reg2 = st.columns([2,3]) # Adjust column widths
    
    with col_reg1:
        st.markdown("#### Summary Statistics by Region")
        regional_stats_df = create_regional_summary(df, selected_year)
        st.dataframe(regional_stats_df, height=300) # Adjust height as needed
    
    with col_reg2:
        st.markdown("#### Regional Comparison (Avg. Life Expectancy & Health Expenditure)")
        # Prepare data for grouped bar chart
        regional_comp_data = regional_stats_df[['life_expectancy', 'health_expenditure_per_capita']].reset_index()
        regional_comp_data = regional_comp_data.melt(id_vars='region', var_name='metric', value_name='value')
        
        regional_bar_chart = alt.Chart(regional_comp_data).mark_bar().encode(
            x=alt.X('region:N', title='Region', sort='-y'),
            y=alt.Y('value:Q', title='Average Value'),
            color=alt.Color('metric:N', title='Metric'),
            column=alt.Column('metric:N', title=None) # Facet by metric
        ).properties(
            width=alt.Step(60), # Adjust bar width
            height=250,
            title=f"Regional Averages for {selected_year}"
        ).resolve_scale(y='independent')
        
        st.altair_chart(regional_bar_chart, use_container_width=True)
    
    st.markdown("---")
    
    # Time series analysis
    st.subheader("📊 Time Series Analysis")
    
    # Country selection for time series
    all_countries_sorted = sorted(df['country'].unique())
    selected_countries_ts = st.multiselect(
        "Select countries to compare over time:",
        options=all_countries_sorted,
        default=['United States', 'China', 'Germany', 'Brazil', 'Nigeria'], # Sensible defaults
        help="Choose countries to show in the time series chart"
    )
    
    if selected_countries_ts:
        time_series_var = st.selectbox(
            "Variable for time series:",
            numeric_vars,
            index=0, # Default to life_expectancy
            key='ts_var_select', # Unique key for widget
            help="Choose which variable to show over time"
        )
        
        time_series_plot = create_time_series_chart(df, selected_countries_ts, time_series_var)
        st.altair_chart(time_series_plot, use_container_width=True)
    else:
        st.info("Select one or more countries to display the time series chart.")
    
    # Data exploration section
    with st.expander("🔍 Explore Raw Data for Selected Year"):
        st.subheader(f"Raw Data Table for {selected_year}")
        st.dataframe(year_data_selected)
        
        # Allow downloading the filtered data
        @st.cache_data # Cache the conversion
        def convert_df_to_csv(input_df):
            return input_df.to_csv(index=False).encode('utf-8')

        csv_download = convert_df_to_csv(year_data_selected)
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv_download,
            file_name=f'gapminder_data_{selected_year}.csv',
            mime='text/csv',
        )

if __name__ == "__main__":
    main()
```

After creating `advanced_gapminder_app.py`, run it:
```bash
streamlit run advanced_gapminder_app.py
```

### **Key Features to Explore in Part 2:**
-   **Dynamic Year Selection:** Use the slider to change the year and see the bubble chart animate (if "Play Animation" is clicked) or update.
-   **Customizable Axes & Size:** Change the X-axis, Y-axis, and bubble size variables to explore different relationships.
-   **Interactive Tooltips:** Hover over bubbles to get detailed country-specific data.
-   **Animation:** Click "Play Animation" to see the chart evolve over the years.
-   **Key Metrics:** See global average metrics update for the selected year.
-   **Regional Analysis:** View summary statistics and comparative bar charts for different regions.
-   **Time Series Comparison:** Select multiple countries and a variable to see trends over time.
-   **Data Exploration:** Expand the "Explore Raw Data" section to view and download the data for the selected year.

This combined demo provides a comprehensive look at Streamlit's capabilities, from simple data views to complex, interactive dashboards.