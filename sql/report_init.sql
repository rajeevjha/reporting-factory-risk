USE CATALOG ${CATALOG};

USE SCHEMA control;
CREATE TABLE IF NOT EXISTS control.risk_rules
(rule_id STRING, name STRING, segment STRING, condition_sql STRING,
 impact_column STRING, impact_value STRING, priority INT, enabled BOOLEAN,
 effective_from DATE, effective_to DATE, owner STRING, notes STRING)
USING DELTA;

-- Assumes bundle uploads files to workspace; keep path symbolic:
COPY INTO control.risk_rules
FROM (SELECT * FROM read_files('${WORKSPACE_ROOT}/control/seed_rules.csv', format => 'csv', header => true));

USE SCHEMA gold;
CREATE TABLE IF NOT EXISTS gold.risk_eval USING DELTA AS
SELECT * FROM ${CATALOG}.gold.features WHERE 1=0;

CREATE TABLE IF NOT EXISTS gold.report_runs
(report_run_id STRING, name STRING, status STRING, rules_version STRING,
 started_at TIMESTAMP, finished_at TIMESTAMP, approved_by STRING, approved_at TIMESTAMP)
USING DELTA;

CREATE TABLE IF NOT EXISTS gold.report_facts
(report_run_id STRING, metric STRING, dimension STRING, value DOUBLE)
USING DELTA;
