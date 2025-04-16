# Demo 2: SQL Basics

In this notebook, we'll learn the fundamentals of SQL queries using our NHANES data.

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

## Basic SELECT

Let's start with simple SELECT statements:

```python
%%sql
-- Select all columns from demographics
SELECT * FROM demographics;
```

```python
%%sql
-- Select specific columns
SELECT seqn, age, gender FROM demographics;
```

```python
%%sql
-- Use column aliases
SELECT 
    seqn AS participant_id,
    age AS participant_age,
    gender AS participant_gender
FROM demographics;
```

## NULL Handling and Error Cases

Let's see how SQL handles NULL values and common error cases:

```python
%%sql
-- NULL handling in aggregates
SELECT 
    gender,
    COUNT(*) as total_count,
    COUNT(education) as non_null_education,
    AVG(age) as avg_age,
    AVG(CASE WHEN age IS NOT NULL THEN age ELSE 0 END) as avg_age_with_null_handling
FROM demographics
GROUP BY gender;
```

```python
%%sql
-- Error case: non-existent column
SELECT non_existent_column FROM demographics;  -- Will show error
```

## WHERE Clause

Practice filtering data:

```python
%%sql
-- Find participants over 50 years old
SELECT * FROM demographics WHERE age > 50;
```

```python
%%sql
-- Find female participants
SELECT * FROM demographics WHERE gender = 'F';
```

```python
%%sql
-- Combine conditions
SELECT * FROM demographics 
WHERE age > 50 AND gender = 'F';
```

```python
%%sql
-- Use IN for multiple values
SELECT * FROM demographics 
WHERE race IN ('White', 'Black');
```

## GROUP BY and Aggregates

Practice grouping and aggregation:

```python
%%sql
-- Count participants by gender
SELECT gender, COUNT(*) AS count
FROM demographics
GROUP BY gender;
```

```python
%%sql
-- Calculate average age by race
SELECT race, AVG(age) AS avg_age
FROM demographics
GROUP BY race;
```

```python
%%sql
-- Multiple aggregates
SELECT 
    race,
    COUNT(*) AS count,
    AVG(age) AS avg_age,
    MIN(age) AS min_age,
    MAX(age) AS max_age
FROM demographics
GROUP BY race;
```

## HAVING Clause

Filter aggregated results:

```python
%%sql
-- Find races with more than 100 participants
SELECT race, COUNT(*) AS count
FROM demographics
GROUP BY race
HAVING COUNT(*) > 100;
```

```python
%%sql
-- Find races with average age over 40
SELECT race, AVG(age) AS avg_age
FROM demographics
GROUP BY race
HAVING AVG(age) > 40;
```

## Practice

Try these exercises:
1. Find the average BMI by gender
2. Count participants by education level
3. Find races with average BMI over 25
4. Calculate the percentage of participants by gender 