USE CATALOG ${CATALOG};
USE SCHEMA gold;

CREATE OR REPLACE TABLE gold.features AS
SELECT
  l.loan_id, l.borrower_id, l.loan_amount, l.interest_rate, l.term, l.grade, l.issue_date,
  b.fico_score, b.dti, b.annual_income, b.utilization,
  CASE WHEN l.loan_amount >= 25000 THEN 'corporate' ELSE 'retail' END AS segment
FROM ${CATALOG}.silver.loans l
LEFT JOIN ${CATALOG}.silver.borrowers b USING (borrower_id);
