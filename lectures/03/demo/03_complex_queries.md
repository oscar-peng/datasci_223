# Demo 3: Complex Queries

In this notebook, we'll explore more advanced SQL concepts using our NHANES data.

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

## JOIN Operations

Practice different types of JOINs:

```python
%%sql
-- INNER JOIN: Find participants with both demographics and lab results
SELECT d.*, l.lbxglu, l.lbxcr
FROM demographics d
INNER JOIN laboratory l ON d.seqn = l.seqn;
```

```python
%%sql
-- LEFT JOIN: Find all participants and their lab results (if any)
SELECT d.*, l.lbxglu, l.lbxcr
FROM demographics d
LEFT JOIN laboratory l ON d.seqn = l.seqn;
```

```python
%%sql
-- Multiple JOINs: Combine demographics, examination, and lab data
SELECT 
    d.seqn, d.age, d.gender,
    e.bmxbmi,
    l.lbxglu, l.lbxcr
FROM demographics d
LEFT JOIN examination e ON d.seqn = e.seqn
LEFT JOIN laboratory l ON d.seqn = l.seqn;
```

## Common JOIN Issues

Let's look at some common JOIN problems:

```python
%%sql
-- Cartesian join example (intentional mistake)
SELECT COUNT(*) FROM demographics, examination;  -- Will show explosion
```

```python
%%sql
-- NULL in joins example
-- First create a "baby" table with some NULL seqn
CREATE TABLE baby AS 
SELECT seqn, age, gender FROM demographics 
WHERE age < 2;

-- Now show how NULLs behave in joins
SELECT 
    b.seqn as baby_seqn,
    d.seqn as demo_seqn,
    b.age as baby_age,
    d.age as demo_age
FROM baby b
LEFT JOIN demographics d ON b.seqn = d.seqn;
```

## Subqueries

Practice using subqueries:

```python
%%sql
-- Find participants with above average BMI
SELECT *
FROM demographics d
JOIN examination e ON d.seqn = e.seqn
WHERE e.bmxbmi > (
    SELECT AVG(bmxbmi) 
    FROM examination
);
```

```python
%%sql
-- Find participants with abnormal lab values
SELECT d.*, l.*
FROM demographics d
JOIN laboratory l ON d.seqn = l.seqn
WHERE l.lbxglu > (
    SELECT AVG(lbxglu) + 2 * STDDEV(lbxglu)
    FROM laboratory
);
```

## Common Table Expressions (CTEs)

Practice using CTEs:

```python
%%sql
-- Calculate age groups using CTE
WITH age_groups AS (
    SELECT 
        seqn,
        CASE 
            WHEN age < 30 THEN '18-29'
            WHEN age < 40 THEN '30-39'
            WHEN age < 50 THEN '40-49'
            ELSE '50+'
        END AS age_group
    FROM demographics
)
SELECT 
    age_group,
    COUNT(*) AS count,
    AVG(bmxbmi) AS avg_bmi
FROM age_groups
JOIN examination ON age_groups.seqn = examination.seqn
GROUP BY age_group;
```

## Window Functions

Practice using window functions:

```python
%%sql
-- Better window function example
SELECT 
    d.seqn,
    d.age,
    e.bmxbmi,
    RANK() OVER (
        PARTITION BY 
            CASE 
                WHEN age < 30 THEN '18-29'
                WHEN age < 40 THEN '30-39'
                WHEN age < 50 THEN '40-49'
                ELSE '50+'
            END
        ORDER BY e.bmxbmi DESC
    ) AS bmi_rank
FROM demographics d
JOIN examination e ON d.seqn = e.seqn;
```

```python
%%sql
-- Calculate running average of lab values
SELECT 
    seqn,
    lbxglu,
    AVG(lbxglu) OVER (ORDER BY seqn ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING) AS running_avg
FROM laboratory;
```

## Practice

Try these exercises:
1. Find participants with both high BMI and abnormal glucose levels
2. Calculate the percentage of participants with diabetes by age group
3. Rank participants by BMI within each race
4. Find participants whose lab values are more than 2 standard deviations from the mean 