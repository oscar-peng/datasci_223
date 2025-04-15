# Demo 2: SQL Basics

## Basic SELECT

Let's start with simple SELECT statements:

```sql
-- Select all columns from demographics
SELECT * FROM demographics;

-- Select specific columns
SELECT seqn, age, gender FROM demographics;

-- Use column aliases
SELECT 
    seqn AS participant_id,
    age AS participant_age,
    gender AS participant_gender
FROM demographics;
```

## WHERE Clause

Practice filtering data:

```sql
-- Find participants over 50 years old
SELECT * FROM demographics WHERE age > 50;

-- Find female participants
SELECT * FROM demographics WHERE gender = 'F';

-- Combine conditions
SELECT * FROM demographics 
WHERE age > 50 AND gender = 'F';

-- Use IN for multiple values
SELECT * FROM demographics 
WHERE race IN ('White', 'Black');
```

## GROUP BY and Aggregates

Practice grouping and aggregation:

```sql
-- Count participants by gender
SELECT gender, COUNT(*) AS count
FROM demographics
GROUP BY gender;

-- Calculate average age by race
SELECT race, AVG(age) AS avg_age
FROM demographics
GROUP BY race;

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

```sql
-- Find races with more than 100 participants
SELECT race, COUNT(*) AS count
FROM demographics
GROUP BY race
HAVING COUNT(*) > 100;

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

## Next Steps
- Preview of complex queries
- Introduction to JOINs 