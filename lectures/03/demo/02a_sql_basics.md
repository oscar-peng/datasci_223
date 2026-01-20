# Demo 2: SQL basics with Chinook

This demo practices core SQL clauses on the Chinook tables imported into DuckDB. Run demo 1 first to set up the database.

## Step 0: Connect to database

```python
%load_ext sql
%sql duckdb:///demo_chinook.duckdb
```

## Step 1: Basic SELECT + LIMIT

```python
%%sql
SELECT TrackId, Name, Milliseconds
FROM tracks
LIMIT 5;
```

Checkpoint: you should see 5 tracks with IDs and durations.

Example output:

| TrackId | Name | Milliseconds |
| --- | --- | --- |
| 1 | For Those About To Rock (We Salute You) | 343719 |
| 2 | Balls to the Wall | 342562 |
| 3 | Fast As a Shark | 230619 |

## Step 2: WHERE + ORDER BY

```python
%%sql
SELECT Name, UnitPrice, Milliseconds
FROM tracks
WHERE UnitPrice >= 0.99
ORDER BY UnitPrice DESC, Milliseconds DESC
LIMIT 10;
```

Checkpoint: higher-priced tracks appear first.

## Step 3: DISTINCT values

```python
%%sql
SELECT DISTINCT Country
FROM customers
ORDER BY Country;
```

Checkpoint: one row per country.

## Step 3b: DISTINCT vs GROUP BY

```python
%%sql
SELECT DISTINCT BillingCountry
FROM invoices
ORDER BY BillingCountry;
```

```python
%%sql
SELECT BillingCountry, COUNT(*) AS invoice_count
FROM invoices
GROUP BY BillingCountry
ORDER BY BillingCountry;
```

Checkpoint: the first query returns unique countries, the second adds counts.

## Step 3c: COUNT(DISTINCT ...)

```python
%%sql
SELECT COUNT(DISTINCT CustomerId) AS unique_customers
FROM invoices;
```

Checkpoint: a single row with a unique customer count.

## Step 3d: CASE WHEN for categories

```python
%%sql
SELECT InvoiceId,
    Total,
    CASE
        WHEN Total >= 10 THEN 'high'
        WHEN Total >= 5 THEN 'medium'
        ELSE 'low'
    END AS spend_bucket
FROM invoices
ORDER BY Total DESC
LIMIT 10;
```

Checkpoint: results include a `spend_bucket` column.

## Step 3e: LIKE vs ILIKE

```python
%%sql
SELECT TrackId, Name
FROM tracks
WHERE Name ILIKE '%love%'
LIMIT 10;
```

Checkpoint: track names containing "love" (case-insensitive).

## Step 4: GROUP BY + aggregates

```python
%%sql
SELECT BillingCountry, COUNT(*) AS invoice_count, ROUND(AVG(Total), 2) AS avg_total
FROM invoices
GROUP BY BillingCountry
ORDER BY invoice_count DESC;
```

Example output:

| BillingCountry | invoice_count | avg_total |
| --- | --- | --- |
| USA | 91 | 5.75 |
| Canada | 56 | 5.42 |
| Brazil | 35 | 5.32 |

## Step 4b: Plot invoice counts by country (Altair)

```python
%%sql invoice_counts <<
SELECT BillingCountry, COUNT(*) AS invoice_count
FROM invoices
GROUP BY BillingCountry
ORDER BY invoice_count DESC
```

```python
import altair as alt

chart = (
    alt.Chart(invoice_counts)
    .mark_bar()
    .encode(
        x=alt.X("BillingCountry:N", sort="-y", title="Billing country"),
        y=alt.Y("invoice_count:Q", title="Invoice count"),
        tooltip=["BillingCountry", "invoice_count"]
    )
)

chart
```

Checkpoint: you should see a bar chart with the largest countries on the left.

## Step 5: HAVING for group filters

```python
%%sql
SELECT BillingCountry, COUNT(*) AS invoice_count
FROM invoices
GROUP BY BillingCountry
HAVING COUNT(*) >= 5
ORDER BY invoice_count DESC;
```

Checkpoint: the result should include only countries with at least 5 invoices.
