-- Run this once in Databricks SQL Editor (choose your SQL Warehouse).
-- It creates helper views your dashboard tiles will query.

USE CATALOG reporting_factory_risk_profile;

CREATE OR REPLACE VIEW gold.report_facts_approved_latest AS
SELECT rf.*
FROM gold.report_facts rf
JOIN (
  SELECT report_run_id
  FROM gold.report_runs
  WHERE status='APPROVED'
  ORDER BY approved_at DESC
  LIMIT 1
) latest USING (report_run_id);

CREATE OR REPLACE VIEW gold.v_kpis_latest AS
SELECT
  MAX(CASE WHEN metric='loans_total'     AND dimension='all' THEN value END) AS loans_total,
  MAX(CASE WHEN metric='high_risk_count' AND dimension='all' THEN value END) AS high_risk_count,
  MAX(CASE WHEN metric='avg_dti'         AND dimension='all' THEN value END) AS avg_dti,
  MAX(CASE WHEN metric='avg_fico'        AND dimension='all' THEN value END) AS avg_fico
FROM gold.report_facts_approved_latest;

CREATE OR REPLACE VIEW gold.v_kpis_history AS
WITH k AS (
  SELECT
    r.report_run_id,
    COALESCE(r.approved_at, r.finished_at, r.started_at) AS approved_at,
    MAX(CASE WHEN f.metric='loans_total'     AND f.dimension='all' THEN f.value END) AS loans_total,
    MAX(CASE WHEN f.metric='high_risk_count' AND f.dimension='all' THEN f.value END) AS high_risk_count,
    MAX(CASE WHEN f.metric='avg_dti'         AND f.dimension='all' THEN f.value END) AS avg_dti,
    MAX(CASE WHEN f.metric='avg_fico'        AND f.dimension='all' THEN f.value END) AS avg_fico
  FROM gold.report_runs r
  JOIN gold.report_facts f USING (report_run_id)
  WHERE r.status='APPROVED'
  GROUP BY r.report_run_id, COALESCE(r.approved_at, r.finished_at, r.started_at)
)
SELECT
  report_run_id,
  approved_at,
  loans_total,
  high_risk_count,
  CASE WHEN loans_total>0 THEN 100.0*high_risk_count/loans_total ELSE NULL END AS pct_high_risk,
  avg_dti,
  avg_fico
FROM k
ORDER BY approved_at;
