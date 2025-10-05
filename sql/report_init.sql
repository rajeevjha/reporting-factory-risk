USE CATALOG reporting_factory_risk_profile;

USE SCHEMA control;
CREATE TABLE IF NOT EXISTS control.risk_rules
(
  rule_id STRING, 
  name STRING, 
  segment STRING, 
  condition_sql STRING,
  impact_column STRING, 
  impact_value STRING, 
  priority INT, 
  enabled BOOLEAN,
  effective_from DATE, 
  effective_to DATE, 
  owner STRING, 
  notes STRING
)
USING DELTA;

COPY INTO control.risk_rules
FROM (
  SELECT
    CAST(rule_id AS STRING)                AS rule_id,
    CAST(name AS STRING)                   AS name,
    CAST(segment AS STRING)                AS segment,
    CAST(condition_sql AS STRING)          AS condition_sql,
    CAST(impact_column AS STRING)          AS impact_column,
    CAST(impact_value AS STRING)           AS impact_value,
    CAST(priority AS INT)                  AS priority,
    -- normalize booleans commonly encoded as 'true'/'false'/'1'/'0'
    CASE 
      WHEN lower(trim(enabled)) IN ('true','1','t','y') THEN true
      ELSE false
    END                                     AS enabled,
    -- parse dates; treat empty as NULL
    NULLIF(to_date(effective_from), date'0001-01-01')  AS effective_from,
    CASE WHEN effective_to IS NULL OR trim(effective_to) = '' 
         THEN NULL ELSE to_date(effective_to) END       AS effective_to,
    CAST(owner AS STRING)                   AS owner,
    CAST(notes AS STRING)                   AS notes
  FROM 'file:/Workspace/Users/rajeev_db@rajeevorganization.onmicrosoft.com/reporting-factory-risk/control/seed_rules.csv'
)
FILEFORMAT = CSV
FORMAT_OPTIONS ('header' = 'true');

USE SCHEMA gold;
CREATE TABLE IF NOT EXISTS gold.risk_eval USING DELTA AS
SELECT * FROM gold.features WHERE 1=0;

CREATE TABLE IF NOT EXISTS gold.report_runs
(
  report_run_id STRING, 
  name STRING, 
  status STRING, 
  rules_version STRING,
  started_at TIMESTAMP, 
  finished_at TIMESTAMP, 
  approved_by STRING, 
  approved_at TIMESTAMP
)
USING DELTA;

CREATE TABLE IF NOT EXISTS gold.report_facts
(
  report_run_id STRING, 
  metric STRING, 
  dimension STRING, 
  value DOUBLE
)
USING DELTA;