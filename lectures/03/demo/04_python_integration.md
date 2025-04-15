# Demo 4: SQL and Python Integration

## Reading Data Directly

Let's start by reading the CSV files directly into pandas:

```python
import pandas as pd

# Read CSV files directly
demographics_df = pd.read_csv('lectures/03/demo/data/demographics.csv')
examination_df = pd.read_csv('lectures/03/demo/data/examination.csv')
laboratory_df = pd.read_csv('lectures/03/demo/data/laboratory.csv')
questionnaire_df = pd.read_csv('lectures/03/demo/data/questionnaire.csv')

# Basic data exploration
print("Demographics shape:", demographics_df.shape)
print("\nFirst few rows of demographics:")
print(demographics_df.head())
```

## SQL Magic with DataFrames

Now let's use SQL magic to query our DataFrames:

```python
# Register DataFrames with DuckDB
con = duckdb.connect(database=':memory:', read_only=False)
con.register('demographics', demographics_df)
con.register('examination', examination_df)
con.register('laboratory', laboratory_df)
con.register('questionnaire', questionnaire_df)

# Use SQL magic
%sql duckdb://localhost

# Query the registered DataFrames
%%sql
SELECT 
    d.age,
    d.gender,
    e.bmxbmi,
    l.lbxglu
FROM demographics d
JOIN examination e ON d.seqn = e.seqn
JOIN laboratory l ON d.seqn = l.seqn
LIMIT 5;
```

## Basic Visualizations

Let's create some simple visualizations:

```python
import matplotlib.pyplot as plt

# Scatter plot of Age vs BMI
plt.figure(figsize=(10, 6))
plt.scatter(demographics_df['age'], examination_df['bmxbmi'], alpha=0.5)
plt.xlabel('Age')
plt.ylabel('BMI')
plt.title('Age vs BMI')
plt.show()

# Box plot of BMI by Gender
plt.figure(figsize=(10, 6))
plt.boxplot([
    examination_df[demographics_df['gender'] == 'M']['bmxbmi'],
    examination_df[demographics_df['gender'] == 'F']['bmxbmi']
])
plt.xticks([1, 2], ['Male', 'Female'])
plt.ylabel('BMI')
plt.title('BMI Distribution by Gender')
plt.show()
```

## Polars Integration (Preview)

For large datasets, Polars can be more efficient. This will be covered in detail in the next lecture:

```python
import polars as pl

# Read CSV with Polars
demographics_pl = pl.read_csv('lectures/03/demo/data/demographics.csv')
examination_pl = pl.read_csv('lectures/03/demo/data/examination.csv')

# Basic operations
print("Demographics shape:", demographics_pl.shape)
print("\nFirst few rows:")
print(demographics_pl.head())
```

## Practice

Try these exercises:
1. Create a scatter plot of BMI vs glucose levels
2. Calculate and visualize the correlation between lab values
3. Create a dashboard of key health metrics by demographic groups
4. Compare performance between pandas and polars for large queries

## Next Steps
- Advanced topics
- Best practices
- Resources for learning 