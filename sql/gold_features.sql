USE CATALOG reporting_factory_risk_profile;
USE SCHEMA gold;

CREATE OR REPLACE TABLE gold.features AS
SELECT
  l.loan_id,
  l.borrower_id,
  b.dti,
  b.fico_score,
  b.utilization,
  l.grade,
  l.loan_amount,
  l.interest_rate,
  l.term_months AS term,      -- ← was l.term
  l.issue_date,
  b.annual_income
FROM silver.loans AS l
JOIN silver.borrowers AS b
  ON l.borrower_id = b.borrower_id;