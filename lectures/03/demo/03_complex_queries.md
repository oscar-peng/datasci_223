# Demo 3: Complex Queries

## JOIN Operations

Practice different types of JOINs:

```sql
-- INNER JOIN: Find participants with both demographics and lab results
SELECT d.*, l.lbxglu, l.lbxcr
FROM demographics d
INNER JOIN laboratory l ON d.seqn = l.seqn;

-- LEFT JOIN: Find all participants and their lab results (if any)
SELECT d.*, l.lbxglu, l.lbxcr
FROM demographics d
LEFT JOIN laboratory l ON d.seqn = l.seqn;

-- Multiple JOINs: Combine demographics, examination, and lab data
SELECT 
    d.seqn, d.age, d.gender,
    e.bmxbmi,
    l.lbxglu, l.lbxcr
FROM demographics d
LEFT JOIN examination e ON d.seqn = e.seqn
LEFT JOIN laboratory l ON d.seqn = l.seqn;
```

## Subqueries

Practice using subqueries:

```sql
-- Find participants with above average BMI
SELECT *
FROM demographics d
JOIN examination e ON d.seqn = e.seqn
WHERE e.bmxbmi > (
    SELECT AVG(bmxbmi) 
    FROM examination
);

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

```sql
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

```sql
-- Rank participants by BMI within age groups
SELECT 
    d.seqn,
    d.age,
    e.bmxbmi,
    RANK() OVER (PARTITION BY d.age/10 ORDER BY e.bmxbmi DESC) AS bmi_rank
FROM demographics d
JOIN examination e ON d.seqn = e.seqn;

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

## Next Steps
- Preview of Python integration
- Data analysis workflows 