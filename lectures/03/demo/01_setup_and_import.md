# Demo 1: Setup and Data Import

## Setup

First, let's install the required packages:

```python
%pip install ipython-sql duckdb pandas polars
```

Now import the necessary libraries and load SQL magic:

```python
import pandas as pd
import duckdb
%load_ext sql
```

## Download NHANES Data

Run the fetch script to download the NHANES data:

```python
%run fetch_nhanes_data.py
```

## Connect to DuckDB

```python
# Connect to DuckDB
con = duckdb.connect(database=':memory:', read_only=False)
%sql duckdb://localhost
```

## Create Tables

Let's create tables for our NHANES data:

```sql
-- Create demographics table
CREATE TABLE demographics (
    seqn INTEGER PRIMARY KEY,
    age INTEGER,
    gender TEXT,
    race TEXT,
    education TEXT
);

-- Create examination table
CREATE TABLE examination (
    seqn INTEGER PRIMARY KEY,
    bmxht REAL,
    bmxwt REAL,
    bmxbmi REAL,
    FOREIGN KEY (seqn) REFERENCES demographics(seqn)
);

-- Create laboratory table
CREATE TABLE laboratory (
    seqn INTEGER PRIMARY KEY,
    lbxglu REAL,
    lbxcr REAL,
    lbxwbc REAL,
    FOREIGN KEY (seqn) REFERENCES demographics(seqn)
);

-- Create questionnaire table
CREATE TABLE questionnaire (
    seqn INTEGER PRIMARY KEY,
    diq010 INTEGER,
    diq050 INTEGER,
    diq160 INTEGER,
    FOREIGN KEY (seqn) REFERENCES demographics(seqn)
);
```

## Import Data

Now let's import the data using COPY:

```sql
-- Import demographics data
COPY demographics FROM 'lectures/03/demo/data/demographics.csv' 
WITH (FORMAT csv, HEADER true, DELIMITER ',');

-- Import examination data
COPY examination FROM 'lectures/03/demo/data/examination.csv' 
WITH (FORMAT csv, HEADER true, DELIMITER ',');

-- Import laboratory data
COPY laboratory FROM 'lectures/03/demo/data/laboratory.csv' 
WITH (FORMAT csv, HEADER true, DELIMITER ',');

-- Import questionnaire data
COPY questionnaire FROM 'lectures/03/demo/data/questionnaire.csv' 
WITH (FORMAT csv, HEADER true, DELIMITER ',');
```

## Verify Data Import

Let's check that our data was imported correctly:

```sql
-- Check table sizes
SELECT 'demographics' AS table_name, COUNT(*) AS row_count FROM demographics
UNION ALL
SELECT 'examination', COUNT(*) FROM examination
UNION ALL
SELECT 'laboratory', COUNT(*) FROM laboratory
UNION ALL
SELECT 'questionnaire', COUNT(*) FROM questionnaire;

-- Check a sample of the data
SELECT * FROM demographics LIMIT 5;
SELECT * FROM examination LIMIT 5;
SELECT * FROM laboratory LIMIT 5;
SELECT * FROM questionnaire LIMIT 5;
```

## Practice

Try these exercises:
1. Create a new table for blood pressure measurements
2. Import the blood pressure data from the examination file
3. Add appropriate foreign key constraints
4. Verify the import was successful 