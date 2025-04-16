# Demo 4: SQL and Python Integration

In this notebook, we'll explore how to integrate SQL with Python for data analysis.

## Setup

First, let's set up our environment:

```python
%pip install jupysql duckdb-engine pandas polars --quiet
import pandas as pd
import duckdb
%load_ext sql
%config SqlMagic.autocommit=True
%config SqlMagic.feedback = False
%config SqlMagic.displaycon = False

# Connect to DuckDB
con = duckdb.connect(database=':memory:', read_only=False)
%sql duckdb:///:memory:
```

## Load Data

Let's load our NHANES data:

```python
%%sql
-- Clean up any existing tables
DROP TABLE IF EXISTS questionnaire;
DROP TABLE IF EXISTS laboratory;
DROP TABLE IF EXISTS examination;
DROP TABLE IF EXISTS demographics;

-- Load demographics
CREATE TABLE demographics AS
SELECT * FROM read_csv_auto('/home/christopher/code/datasci_223/data/demographic.csv');

-- Load examination
CREATE TABLE examination AS
SELECT * FROM read_csv_auto('/home/christopher/code/datasci_223/data/examination.csv');

-- Load laboratory
CREATE TABLE laboratory AS
SELECT * FROM read_csv_auto('/home/christopher/code/datasci_223/data/labs.csv');

-- Load questionnaire
CREATE TABLE questionnaire AS
SELECT * FROM read_csv_auto('/home/christopher/code/datasci_223/data/questionnaire.csv');
```

## SQL Magic

Practice using SQL magic commands:

```python
# Basic SQL query
%sql SELECT * FROM demographics LIMIT 5;
```

```python
%%sql
SELECT 
    d.age,
    AVG(e.bmxbmi) AS avg_bmi
FROM demographics d
JOIN examination e ON d.seqn = e.seqn
GROUP BY d.age
ORDER BY d.age;
```

## Pandas Integration

Practice SQL to DataFrame conversion:

```python
# Query to DataFrame
df = %sql SELECT * FROM demographics
df.head()
```

```python
# Complex query to DataFrame
bmi_df = %sql SELECT d.*, e.bmxbmi FROM demographics d JOIN examination e ON d.seqn = e.seqn
bmi_df.head()
```

```python
# Analyze data in pandas
bmi_by_age = bmi_df.groupby('age')['bmxbmi'].mean()
bmi_by_age.plot(kind='bar')
```

## Polars Integration

Practice using Polars for high-performance queries:

```python
import polars as pl

# Query to Polars DataFrame
polars_df = pl.read_sql("""
    SELECT 
        d.age,
        d.gender,
        e.bmxbmi,
        l.lbxglu
    FROM demographics d
    JOIN examination e ON d.seqn = e.seqn
    JOIN laboratory l ON d.seqn = l.seqn
""", con)

# Fast aggregations
polars_df.groupby('age').agg([
    pl.col('bmxbmi').mean().alias('avg_bmi'),
    pl.col('lbxglu').mean().alias('avg_glucose')
])
```

## Visualization

Create visualizations from SQL results:

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Get data for visualization
viz_df = %sql SELECT d.age, d.gender, e.bmxbmi FROM demographics d JOIN examination e ON d.seqn = e.seqn

# Create plots
plt.figure(figsize=(10, 6))
sns.boxplot(data=viz_df, x='age', y='bmxbmi', hue='gender')
plt.title('BMI Distribution by Age and Gender')
plt.show()
```

## Practice

Try these exercises:
1. Create a scatter plot of BMI vs glucose levels
2. Calculate and visualize the correlation between lab values
3. Create a dashboard of key health metrics by demographic groups
4. Compare performance between pandas and polars for large queries 