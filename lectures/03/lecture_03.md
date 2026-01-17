---
lecture_number: 03
pdf: true
---

03: SQL for Health Data

- hw03 - #FIXME add GitHub Classroom link once ready

# Links & Self-Guided Review

- [DuckDB docs](https://duckdb.org/docs/) - embedded analytics database with strong CSV support
- [JupySQL](https://jupysql.ploomber.io/) - SQL magics for notebooks
- [SQL Style Guide (Mode)](https://mode.com/sql-tutorial/sql-style-guide/) - readable query conventions
- [SQLite docs](https://www.sqlite.org/docs.html) - lightweight SQL reference
- [PostgreSQL docs](https://www.postgresql.org/docs/) - production SQL dialect used widely in health systems

# Outline

- Why SQL still runs the data world
- SQL in notebooks (JupySQL + DuckDB/SQLite)
- SQL basics: comments, NULLs, SELECT/WHERE
- Live demo 1
- SQL statement structure and execution order
- Importing data and defining tables
- Filtering and aggregation (WHERE vs HAVING)
- Joins, subqueries, CTEs, window functions
- Views, performance basics, SQL + Python workflows
- Live demo 2
- Live demo 3

# Why SQL still runs the data world

SQL is how data teams ask precise questions of large relational datasets. In health data science, it shows up everywhere: EHR extracts, claims data, public health reporting, and analytics layers under dashboards.

| Where SQL shows up | What you use it for |
| --- | --- |
| EHR reporting | Cohorts, encounters, lab results |
| Claims data | Cost summaries, utilization, billing audits |
| Research datasets | Cleaning, joining, aggregating |
| Analytics stacks | ELT pipelines, dashboards, scheduled reports |

![XKCD: Data Point](03/media/xkcd_data_point.png)

### Reference Card: Why SQL matters

| Strength | Why it helps in practice |
| --- | --- |
| Declarative | You state what you want, not how to compute it |
| Scalable | Works on datasets larger than laptop RAM |
| Transferable | Similar syntax across DuckDB, SQLite, PostgreSQL |
| Composable | Queries can be layered and reused |

### Code Snippet: Smallest useful query

```sql
SELECT patient_id, age, sex
FROM demographics
LIMIT 5;
```

# SQL in notebooks (JupySQL + DuckDB/SQLite)

DuckDB is an embedded analytics database that feels like SQLite but is optimized for analytics. JupySQL lets you run SQL directly in notebooks using `%sql` for single-line and `%%sql` for multi-line queries, and it works with any SQLAlchemy-supported engine.

#FIXME: add a visual showing SQL to pandas flow

### Reference Card: Notebook setup

| Step | Command |
| --- | --- |
| Install | `pip install duckdb duckdb-engine jupysql pandas` |
| Load magic | `%load_ext sql` |
| Connect (DuckDB) | `%sql duckdb:///clinic.db` |
| Connect (SQLite) | `%sql sqlite:///clinic.db` |
| Return DataFrame | `result = %sql SELECT * FROM table` |

### Code Snippet: Minimal setup

```python
import duckdb

%load_ext sql
%sql duckdb:///clinic.db

%%sql
SELECT * FROM demographics
LIMIT 5;
```

# SQL basics: comments, semicolons, and NULLs

SQL statements end with semicolons, and comments start with `--`. NULL means missing, so comparisons need `IS NULL` rather than `=`.

| Value | Interpretation | Correct check |
| --- | --- | --- |
| `NULL` | Missing value | `IS NULL` / `IS NOT NULL` |
| Empty string | Present but blank | `= ''` |
| Zero | Numeric value | `= 0` |

### Reference Card: NULL handling

| Pattern | Example | Notes |
| --- | --- | --- |
| Check missing | `WHERE lab_value IS NULL` | NULL is not equal to NULL |
| Fill missing | `COALESCE(lab_value, 0)` | Replace NULL with default |
| Safer filters | `WHERE lab_value IS NOT NULL` | Avoid accidental drops |

### Code Snippet: NULL safe filtering

```sql
SELECT patient_id, test_name, value
FROM labs
WHERE value IS NOT NULL
ORDER BY test_name;
```

# SQL basics: SELECT, WHERE, ORDER BY, LIMIT, DISTINCT

This is the core of most queries: choose columns, filter rows, and order the output. Learn these first and most day-to-day SQL becomes readable.

| patient_id | encounter_date | department | total_cost |
| --- | --- | --- | --- |
| 2009 | 2024-03-06 | Cardiology | 560.00 |
| 2006 | 2024-03-01 | Emergency | 980.00 |
| 2010 | 2024-02-28 | Emergency | 1500.00 |

### Reference Card: Core clauses

| Clause | Purpose | Example |
| --- | --- | --- |
| `SELECT` | Choose columns | `SELECT patient_id, age` |
| `WHERE` | Filter rows | `WHERE age >= 18` |
| `ORDER BY` | Sort results | `ORDER BY total_cost DESC` |
| `LIMIT` | Keep top N | `LIMIT 10` |
| `DISTINCT` | Unique values | `SELECT DISTINCT department` |

### Reference Card: WHERE operators

| Operator | Purpose | Example |
| --- | --- | --- |
| Comparisons | Match values | `age >= 18` |
| Logical | Combine conditions | `age >= 18 AND sex = 'F'` |
| `IN` | Match a set | `department IN ('ER', 'ICU')` |
| `BETWEEN` | Match ranges | `total_cost BETWEEN 100 AND 500` |
| `LIKE` | Pattern match | `test_name LIKE '%A1C%'` |

### Code Snippet: Basic filtering

```sql
SELECT encounter_id, patient_id, department, total_cost
FROM encounters
WHERE total_cost >= 500
ORDER BY total_cost DESC
LIMIT 5;
```

# LIVE DEMO!

# SQL statement structure and execution order

SQL reads like English, but it executes in a specific order. When a query is confusing, rewrite it as a pipeline: join tables, filter rows, group, compute, then sort.

#FIXME: add execution order diagram

### Reference Card: Execution order (simplified)

| Order | Clause |
| --- | --- |
| 1 | `FROM` + `JOIN` |
| 2 | `WHERE` |
| 3 | `GROUP BY` |
| 4 | `HAVING` |
| 5 | `SELECT` |
| 6 | `ORDER BY` |
| 7 | `LIMIT` |

### Code Snippet: Readable query template

```sql
SELECT department, COUNT(*) AS visit_count
FROM encounters
WHERE total_cost > 500
GROUP BY department
HAVING COUNT(*) >= 3
ORDER BY visit_count DESC
LIMIT 5;
```

# Importing data and defining tables

Real projects start with files. You can load CSVs directly, scan Parquet for columnar files, or create tables with explicit types for safer analysis.

#FIXME: add data import diagram or screenshot

### Reference Card: Common import patterns

| Pattern | Use case | Example |
| --- | --- | --- |
| `read_csv_auto` | Quick CSV exploration | `SELECT * FROM read_csv_auto('file.csv')` |
| `read_parquet` | Fast columnar reads | `SELECT * FROM read_parquet('file.parquet')` |
| `CREATE TABLE AS` | Persist results | `CREATE TABLE t AS SELECT ...` |
| `COPY` | Fast bulk load | `COPY t FROM 'file.csv' (HEADER true)` |

### Code Snippet: Create table from CSV

```sql
CREATE TABLE demographics AS
SELECT * FROM read_csv_auto('demographics.csv');
```

# Filtering and aggregation

Filtering shrinks the data to what you need. Aggregation summarizes it to the level you want to report. Use `WHERE` to filter rows before grouping and `HAVING` to filter groups after aggregation; if you are unsure, start with `WHERE`.

#FIXME: add a WHERE vs HAVING diagram

| Department | Visit count | Avg cost |
| --- | --- | --- |
| Cardiology | 7 | 812.50 |
| Oncology | 5 | 1230.20 |
| Primary Care | 12 | 210.30 |

### Reference Card: Filtering + aggregates

| Tool | Example | Purpose |
| --- | --- | --- |
| `GROUP BY` | `GROUP BY department` | Define groups |
| `HAVING` | `HAVING COUNT(*) > 3` | Filter groups |
| Aggregates | `COUNT`, `AVG`, `SUM` | Summarize |

### Code Snippet: Filter then group

```sql
SELECT department,
    COUNT(*) AS visit_count,
    AVG(total_cost) AS avg_cost
FROM encounters
WHERE encounter_date >= '2024-01-01'
GROUP BY department
ORDER BY avg_cost DESC;
```

![XKCD: Selection Bias](03/media/xkcd_selection_bias.png)

# Joins: combine tables safely

Joins connect demographics, encounters, and lab results. Always join on keys and check for row explosion (many-to-many joins can multiply rows).

#FIXME: add join types diagram

### Reference Card: Join types

| Join | Keeps rows from | When to use |
| --- | --- | --- |
| `INNER JOIN` | Both tables | Only matched records |
| `LEFT JOIN` | Left table | Keep all patients, add matches |
| `RIGHT JOIN` | Right table | Rare in analytics |
| `FULL JOIN` | Both tables | Audits, reconciliation |

### Code Snippet: Encounter + demographics

```sql
SELECT e.encounter_id, e.department, d.age, d.sex
FROM encounters AS e
LEFT JOIN demographics AS d
    ON e.patient_id = d.patient_id;
```

# LIVE DEMO!!

# Subqueries and CTEs

Subqueries let you nest logic. CTEs (`WITH`) make that logic readable and reusable, which matters for multi-step cohort definitions.

#FIXME: add subquery vs CTE comparison visual

### Reference Card: Subqueries vs CTEs

| Pattern | Best for | Example |
| --- | --- | --- |
| Subquery | Small, single use logic | `WHERE id IN (SELECT ...)` |
| CTE | Multi-step, readable logic | `WITH cohort AS (...)` |

### Code Snippet: CTE for a cohort

```sql
WITH high_a1c AS (
    SELECT patient_id
    FROM labs
    WHERE test_name = 'A1C' AND value >= 7.0
)
SELECT d.patient_id, d.age, d.sex
FROM demographics AS d
INNER JOIN high_a1c AS h
    ON d.patient_id = h.patient_id;
```

# Window functions

Window functions compute metrics across related rows without collapsing them. They are common in claims and encounter analytics (running totals, most recent visit, rank within a patient).

| patient_id | encounter_date | total_cost | running_cost |
| --- | --- | --- | --- |
| 101 | 2024-01-10 | 220.00 | 220.00 |
| 101 | 2024-02-03 | 125.00 | 345.00 |
| 101 | 2024-03-02 | 410.00 | 755.00 |

### Reference Card: Window function patterns

| Function | Use case | Example |
| --- | --- | --- |
| `ROW_NUMBER` | Order rows | `ROW_NUMBER() OVER (...)` |
| `LAG` | Compare to previous | `LAG(total_cost)` |
| `SUM` | Running totals | `SUM(total_cost) OVER (...)` |

### Code Snippet: Running total per patient

```sql
SELECT patient_id,
    encounter_date,
    total_cost,
    SUM(total_cost) OVER (
        PARTITION BY patient_id
        ORDER BY encounter_date
    ) AS running_cost
FROM encounters
ORDER BY patient_id, encounter_date;
```

![XKCD: Proxy Variable](03/media/xkcd_proxy_variable.png)

# Data modification (INSERT, UPDATE, DELETE)

These statements change data in place. Use them sparingly in analytics and favor scratch databases when you need to write.

#FIXME: add data modification workflow visual

### Reference Card: Data modification statements

| Statement | Use case | Guardrail |
| --- | --- | --- |
| `INSERT` | Add rows | Use explicit column lists |
| `UPDATE` | Edit rows | Always include a `WHERE` |
| `DELETE` | Remove rows | Preview with `SELECT` first |

### Code Snippet: Safe insert

```sql
INSERT INTO solution (id, name)
VALUES (1, 'Ada Lovelace');
```

# Views and materialized views

Views save a query definition. Materialized views store a cached snapshot of results, which can speed up dashboards at the cost of refresh time.

#FIXME: add view vs materialized view visual

### Reference Card: Views

| Type | Stored data? | Use case |
| --- | --- | --- |
| View | No | Reusable logic, always current |
| Materialized view | Yes | Faster reads, scheduled refresh |

### Code Snippet: Create a view

```sql
CREATE VIEW high_cost_encounters AS
SELECT encounter_id, patient_id, total_cost
FROM encounters
WHERE total_cost >= 1000;
```

# Performance basics

Performance depends on reading less data and doing work once. A query plan is the database's step-by-step execution order; start with filters and clear joins before reaching for complex optimizations.

#FIXME: add query plan or performance visual

![XKCD: Complexity Analysis](03/media/xkcd_complexity_analysis.png)

### Reference Card: Performance checklist

| Tip | Why it helps |
| --- | --- |
| Filter early | Less data to join and aggregate |
| Select only needed columns | Reduces memory usage |
| Use indexes (when supported) | Faster lookups |
| Inspect query plans | Find slow steps |

### Code Snippet: Explain a query

```sql
EXPLAIN
SELECT department, COUNT(*)
FROM encounters
GROUP BY department;
```

# SQL + Python workflows

SQL is great for filtering and aggregation; Python is great for modeling and visualization. Use SQL to reduce the dataset, then load the result into pandas for analysis. DuckDB and SQLite are easy local options; for external databases, connect via SQLAlchemy and reuse the same query strings.

| Step | Tool | Output |
| --- | --- | --- |
| Query | SQL | Table of filtered rows |
| Analyze | pandas | DataFrame, plots |
| Model | sklearn | Features, predictions |

### Reference Card: pandas interop

| Method | Usage |
| --- | --- |
| `pd.read_sql` | `pd.read_sql(query, connection)` |
| DuckDB to pandas | `duckdb.sql(query).df()` |
| Save results | `df.to_parquet("outputs/report.parquet")` |

### Code Snippet: SQLAlchemy connection

```python
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql://user:password@localhost:5432/clinic")
query = "SELECT department, COUNT(*) AS visit_count FROM encounters GROUP BY department"
report = pd.read_sql(query, engine)
```

### Code Snippet: SQL to DataFrame

```python
import duckdb
import pandas as pd

con = duckdb.connect("clinic.db")
query = """
SELECT department, COUNT(*) AS visit_count
FROM encounters
GROUP BY department
"""
report = con.execute(query).fetch_df()
print(report.head())
```

# LIVE DEMO!!!
