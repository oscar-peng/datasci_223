# Demo 3: Joins, CTEs, windows, and performance

This demo combines multiple tables and shows patterns used in real analytics work. Run demo 1 first to set up the database.

## Step 0: Connect to database

```python
%load_ext sql
%sql duckdb:///demo_chinook.duckdb
```

## Step 1: Join tracks to artists

```python
%%sql
SELECT t.Name AS track_name,
    a.Name AS artist_name,
    t.UnitPrice
FROM tracks AS t
INNER JOIN albums AS al
    ON t.AlbumId = al.AlbumId
INNER JOIN artists AS a
    ON al.ArtistId = a.ArtistId
ORDER BY t.UnitPrice DESC, t.Name
LIMIT 10;
```

Checkpoint: results should show track names alongside artist names.

Example output:

| track_name | artist_name | UnitPrice |
| --- | --- | --- |
| The Woman King | Iron Maiden | 1.99 |
| The Trooper | Iron Maiden | 1.99 |

## Step 2: CTE to define a cohort of high-value customers

```python
%%sql
WITH customer_spend AS (
    SELECT CustomerId, SUM(Total) AS total_spend
    FROM invoices
    GROUP BY CustomerId
),
high_value AS (
    SELECT CustomerId
    FROM customer_spend
    WHERE total_spend >= 40
)
SELECT c.FirstName, c.LastName, c.Country, cs.total_spend
FROM customers AS c
INNER JOIN customer_spend AS cs
    ON c.CustomerId = cs.CustomerId
INNER JOIN high_value AS h
    ON c.CustomerId = h.CustomerId
ORDER BY cs.total_spend DESC;
```

Checkpoint: results should show only customers with total spend >= 40.

Example output:

| FirstName | LastName | Country | total_spend |
| --- | --- | --- | --- |
| Victor | Stevens | USA | 49.62 |
| Astrid | Gruber | Germany | 42.62 |

## Step 2b: IN vs EXISTS

```python
%%sql
SELECT c.CustomerId, c.FirstName, c.LastName
FROM customers AS c
WHERE c.CustomerId IN (
    SELECT CustomerId
    FROM invoices
);
```

```python
%%sql
SELECT c.CustomerId, c.FirstName, c.LastName
FROM customers AS c
WHERE EXISTS (
    SELECT 1
    FROM invoices AS i
    WHERE i.CustomerId = c.CustomerId
);
```

Checkpoint: both queries return the same customers.

## Step 3: Window function for running totals

```python
%%sql
SELECT InvoiceId,
    InvoiceDate,
    Total,
    SUM(Total) OVER (
        ORDER BY InvoiceDate
    ) AS running_total
FROM invoices
ORDER BY InvoiceDate
LIMIT 20;
```

Checkpoint: `running_total` should increase over time.

## Step 3b: Plot running totals (Altair)

```python
%%sql running_totals <<
SELECT InvoiceDate, Total,
    SUM(Total) OVER (ORDER BY InvoiceDate) AS running_total
FROM invoices
ORDER BY InvoiceDate
```

```python
import altair as alt

chart = (
    alt.Chart(running_totals)
    .mark_line()
    .encode(
        x=alt.X("InvoiceDate:T", title="Invoice date"),
        y=alt.Y("running_total:Q", title="Running total"),
        tooltip=["InvoiceDate", "running_total"]
    )
)

chart
```

Checkpoint: the line should increase over time.

## Step 4: Create a view for reuse

```python
%%sql
DROP VIEW IF EXISTS top_tracks;

CREATE VIEW top_tracks AS
SELECT t.TrackId, t.Name, t.UnitPrice
FROM tracks AS t
ORDER BY t.UnitPrice DESC;
```

```python
%%sql
SELECT * FROM top_tracks LIMIT 5;
```

Checkpoint: results should show the most expensive tracks.

## Step 5: Inspect a query plan

```python
%%sql
EXPLAIN
SELECT BillingCountry, COUNT(*) AS invoice_count
FROM invoices
GROUP BY BillingCountry;
```

Checkpoint: output should be a query plan (not a regular result table).

## Step 6: DATE_TRUNC for time buckets

```python
%%sql
SELECT DATE_TRUNC('month', InvoiceDate) AS month, COUNT(*) AS invoice_count
FROM invoices
GROUP BY month
ORDER BY month;
```

Checkpoint: one row per month bucket.
