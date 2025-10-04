USE CATALOG ${CATALOG};
USE SCHEMA silver;

CREATE OR REPLACE TABLE silver.loans AS
SELECT DISTINCT
  CAST(id AS STRING) AS loan_id,
  CAST(member_id AS STRING) AS borrower_id,
  CAST(loan_amnt AS DOUBLE) AS loan_amount,
  CAST(int_rate AS DOUBLE) AS interest_rate,
  CAST(term AS STRING) AS term,
  CAST(grade AS STRING) AS grade,
  CAST(issue_d AS DATE) AS issue_date
FROM ${CATALOG}.bronze.lending_raw
WHERE id IS NOT NULL;

CREATE OR REPLACE TABLE silver.borrowers AS
SELECT DISTINCT
  CAST(member_id AS STRING) AS borrower_id,
  CAST(dti AS DOUBLE) AS dti,
  CAST(annual_inc AS DOUBLE) AS annual_income,
  CAST(revol_util AS DOUBLE) AS utilization,
  CAST(fico_range_high AS INT) AS fico_score
FROM ${CATALOG}.bronze.lending_raw
WHERE member_id IS NOT NULL;
