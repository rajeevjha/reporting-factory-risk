import streamlit as st
import databricks.sql as sql
from databricks.sdk.core import Config
import os
import pandas as pd
import textwrap
import time
from datetime import datetime, timezone

# ────────────────────────────────────────────────
# Streamlit page config  (must be first Streamlit call)
# ────────────────────────────────────────────────
st.set_page_config(page_title="Risk Rules Editor", layout="wide")
st.title("💼 Risk Profiling — Business Rules Editor")

# ────────────────────────────────────────────────
# Databricks SQL connection
# ────────────────────────────────────────────────
cfg = Config()
DATABRICKS_SERVER   = cfg.host or os.getenv("DATABRICKS_HOST")
DATABRICKS_HTTPPATH = os.getenv("DATABRICKS_HTTP_PATH","/sql/1.0/warehouses/5aed87c2fa2d4a3f")
CATALOG             = os.getenv("DBSQL_CATALOG", "reporting_factory_risk_profile")

_conn = None

def get_conn():
    """Return or reopen Databricks SQL connection."""
    from databricks import sql
    global _conn
    try:
        if _conn is None:
            _conn = sql.connect(
                server_hostname=DATABRICKS_SERVER,
                http_path=DATABRICKS_HTTPPATH,
                credentials_provider=lambda: cfg.authenticate,
            )
        cur = _conn.cursor()
        cur.close()
        return _conn
    except Exception:
        try:
            if _conn:
                _conn.close()
        finally:
            _conn = sql.connect(
                server_hostname=DATABRICKS_SERVER,
                http_path=DATABRICKS_HTTPPATH,
                credentials_provider=lambda: cfg.authenticate,
            )
        return _conn

def run_one(sql_text: str) -> pd.DataFrame:
    """Execute a single SQL and return DataFrame."""
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(sql_text)
        if cur.description:
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
            return pd.DataFrame.from_records(rows, columns=cols)
        return pd.DataFrame()

def exec_sql(sql_text: str):
    """Execute one or more SQL statements (no result)."""
    conn = get_conn()
    with conn.cursor() as cur:
        for stmt in [s.strip() for s in sql_text.split(";") if s.strip()]:
            cur.execute(stmt)

# ────────────────────────────────────────────────
# Load current rules
# ────────────────────────────────────────────────
exec_sql(f"USE CATALOG {CATALOG}")
df_rules = run_one("""
SELECT rule_id, name, segment, condition_sql, impact_column, impact_value,
       priority, enabled, effective_from, effective_to, owner, notes
FROM control.risk_rules
ORDER BY priority DESC, rule_id
""")

st.subheader("✏️ Inline Edit or Add Business Rules")
st.caption("Edit directly in the grid, or add a new row at the bottom.")

edited = st.data_editor(
    df_rules,
    num_rows="dynamic",
    use_container_width=True,
    key="rules_editor",
    height=420,
)

col1, col2 = st.columns([1, 1])
with col1:
    save_btn = st.button("💾 Save Changes", type="primary")
with col2:
    reevaluate = st.button("🔁 Re-Evaluate Risk")

# ────────────────────────────────────────────────
# Save updates
# ────────────────────────────────────────────────
def sql_literal(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "NULL"
    s = str(x).replace("'", "''")
    return f"'{s}'"

def upsert_rows_param(rows):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(f"USE CATALOG {CATALOG}")
        for _, r in rows.iterrows():
            # normalize types
            priority = None if pd.isna(r.get("priority")) else int(r["priority"])
            enabled  = str(r.get("enabled")).lower() in ("true","1","t","yes","y")
            # date -> 'YYYY-MM-DD' or None
            ef = r.get("effective_from")
            et = r.get("effective_to")
            ef = None if (ef is None or (isinstance(ef, float) and pd.isna(ef))) else str(ef)[:10]
            et = None if (et is None or (isinstance(et, float) and pd.isna(et))) else str(et)[:10]

            # 1) UPDATE by rule_id
            cur.execute("""
                UPDATE control.risk_rules
                SET name=?, segment=?, condition_sql=?, impact_column=?, impact_value=?,
                    priority=?, enabled=?, effective_from=?, effective_to=?, owner=?, notes=?
                WHERE rule_id=?""",
                (
                  None if pd.isna(r.get("name")) else str(r["name"]),
                  None if pd.isna(r.get("segment")) else str(r["segment"]),
                  None if pd.isna(r.get("condition_sql")) else str(r["condition_sql"]),
                  None if pd.isna(r.get("impact_column")) else str(r["impact_column"]),
                  None if pd.isna(r.get("impact_value")) else str(r["impact_value"]),
                  priority, enabled, ef, et,
                  None if pd.isna(r.get("owner")) else str(r["owner"]),
                  None if pd.isna(r.get("notes")) else str(r["notes"]),
                  str(r["rule_id"]),
                ),
            )
            # 2) If not updated, INSERT
            if cur.rowcount == 0:
                cur.execute("""
                    INSERT INTO control.risk_rules
                    (rule_id, name, segment, condition_sql, impact_column, impact_value,
                     priority, enabled, effective_from, effective_to, owner, notes)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                      str(r["rule_id"]),
                      None if pd.isna(r.get("name")) else str(r["name"]),
                      None if pd.isna(r.get("segment")) else str(r["segment"]),
                      None if pd.isna(r.get("condition_sql")) else str(r["condition_sql"]),
                      None if pd.isna(r.get("impact_column")) else str(r["impact_column"]),
                      None if pd.isna(r.get("impact_value")) else str(r["impact_value"]),
                      priority, enabled, ef, et,
                      None if pd.isna(r.get("owner")) else str(r["owner"]),
                      None if pd.isna(r.get("notes")) else str(r["notes"]),
                    ),
                )

if save_btn:
    upsert_rows_param(edited)
    st.success("✅ Rules saved (quotes preserved).")
    st.rerun()

# ────────────────────────────────────────────────
# Re-evaluate risk and show KPIs
# ────────────────────────────────────────────────
if reevaluate:
    exec_sql(f"USE CATALOG {CATALOG}")

    rules_df = run_one("""
SELECT rule_id, impact_column, impact_value, priority, condition_sql
FROM control.risk_rules
WHERE enabled = TRUE
  AND current_date BETWEEN effective_from AND COALESCE(effective_to, DATE '2999-12-31')
ORDER BY priority
""")

    if rules_df.empty:
        st.warning("No enabled rules found. All loans will default to Medium / 0.")
        matches_sql = "SELECT NULL AS loan_id, NULL AS rule_id, NULL AS impact_column, NULL AS impact_value, 0 AS priority WHERE 1=0"
    else:
        parts = []
        for _, r in rules_df.iterrows():
            cond = r["condition_sql"] or "1=0"
            parts.append(f"""
SELECT f.loan_id,
       '{r['rule_id']}' AS rule_id,
       '{r['impact_column']}' AS impact_column,
       '{r['impact_value']}' AS impact_value,
       {int(r['priority']) if pd.notna(r['priority']) else 0} AS priority
FROM {CATALOG}.gold.features f
WHERE ({cond})
""")
        matches_sql = " UNION ALL ".join(parts)

    # Persist gold.risk_eval safely
    exec_sql(f"""
CREATE OR REPLACE TABLE gold.risk_eval
USING DELTA AS
WITH matches AS ({matches_sql}),
band AS (
  SELECT loan_id, impact_value AS risk_band
  FROM (
    SELECT loan_id, impact_value, priority,
           ROW_NUMBER() OVER (PARTITION BY loan_id ORDER BY priority DESC) rn
    FROM matches
    WHERE lower(impact_column)='risk_band'
  ) x WHERE rn=1
),
points AS (
  SELECT loan_id,
         COALESCE(SUM(COALESCE(TRY_CAST(regexp_extract(impact_value,'[-+]?\\d+',0) AS INT),0)),0) AS risk_points
  FROM matches
  WHERE lower(impact_column)='risk_points'
  GROUP BY loan_id
)
SELECT
  f.*,
  COALESCE(band.risk_band,'Medium') AS risk_band,
  COALESCE(points.risk_points,0)    AS risk_points,
  current_timestamp()               AS evaluated_at
FROM gold.features f
LEFT JOIN band   USING (loan_id)
LEFT JOIN points USING (loan_id);
""")

st.success("✅ Risk evaluation updated (gold.risk_eval).")

# KPIs preview
exec_sql(f"USE CATALOG {CATALOG}")
kpis = run_one("""
SELECT
  COUNT(*) AS loans_total,
  SUM(CASE WHEN risk_band='High' OR risk_points>=20 THEN 1 ELSE 0 END) AS high_risk_count,
  COALESCE(ROUND(AVG(CASE 
                        WHEN risk_band='High' OR risk_points>=20 THEN dti 
                      END), 1), 0) AS avg_dti,
  COALESCE(ROUND(AVG(CASE WHEN risk_band='High' OR risk_points>=20 THEN fico_score END), 0), 0) 
  AS avg_fico
FROM gold.risk_eval
""")
st.subheader("📊 Current Risk KPIs")
st.dataframe(kpis, use_container_width=True)

st.caption("Edit rules → Save → Re-Evaluate → KPIs update instantly.")



# ────────────────────────────────────────────────
# Runs History
# ────────────────────────────────────────────────
st.subheader("🕘 Report Runs History")

exec_sql(f"USE CATALOG {CATALOG}")
runs = run_one("""
SELECT
  report_run_id,
  name,
  status,
  rules_version,
  started_at,
  finished_at,
  approved_by,
  approved_at
FROM gold.report_runs
ORDER BY COALESCE(approved_at, finished_at, started_at) DESC
LIMIT 12
""")
st.dataframe(runs, use_container_width=True)

from datetime import datetime, timezone

st.subheader("📝 Create Draft from current risk_eval")

draft_name = st.text_input("Draft name", value="Risk Report (Draft)")
if st.button("Save as Draft"):
    exec_sql(f"USE CATALOG {CATALOG}")

    cnt = run_one("SELECT COUNT(*) AS c FROM gold.risk_eval")
    if cnt.empty or int(cnt.iloc[0]["c"]) == 0:
        st.error("No rows in gold.risk_eval. Re-evaluate first, then create draft.")
        st.stop()

    draft_id = f"RR_DRAFT_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    # Insert draft header
    exec_sql(f"""
INSERT INTO gold.report_runs
(report_run_id, name, status, rules_version, started_at, finished_at, approved_by, approved_at)
VALUES ('{draft_id}', {sql_literal(draft_name)}, 'DRAFT', 'ui@app', current_timestamp(), current_timestamp(), NULL, NULL);
""")

    # Insert draft facts from current risk_eval
    exec_sql(f"""
INSERT INTO gold.report_facts (report_run_id, metric, dimension, value)
SELECT '{draft_id}','loans_total','all', CAST(COUNT(*) AS DOUBLE) FROM gold.risk_eval;

INSERT INTO gold.report_facts (report_run_id, metric, dimension, value)
SELECT '{draft_id}','high_risk_count','all',
       CAST(SUM(CASE WHEN risk_band='High' OR risk_points>=20 THEN 1 ELSE 0 END) AS DOUBLE)
FROM gold.risk_eval;

INSERT INTO gold.report_facts (report_run_id, metric, dimension, value)
SELECT '{draft_id}','avg_dti','all',  CAST(AVG(dti) AS DOUBLE) FROM gold.risk_eval;

INSERT INTO gold.report_facts (report_run_id, metric, dimension, value)
SELECT '{draft_id}','avg_fico','all', CAST(AVG(fico_score) AS DOUBLE) FROM gold.risk_eval;
""")

    st.success(f"Draft created: {draft_id}")
    st.rerun()

    st.subheader("🚀 Promote Draft → Approved")

# Load drafts for selection
exec_sql(f"USE CATALOG {CATALOG}")
drafts = run_one("""
SELECT report_run_id, name, started_at
FROM gold.report_runs
WHERE status='DRAFT'
ORDER BY started_at DESC
LIMIT 50
""")

if drafts.empty:
    st.info("No drafts available. Create a draft first.")
else:
    options = [f"{r['report_run_id']}  —  {r['name']}  ({r['started_at']})" for _, r in drafts.iterrows()]
    sel = st.selectbox("Select a draft to approve", options, index=0)
    selected_run_id = sel.split("—")[0].strip()

    approved_by = st.text_input("Approved by (optional)", value="")
    if st.button("✅ Approve selected draft"):
        exec_sql(f"USE CATALOG {CATALOG}")
        # Mark run approved (facts already exist for that run_id)
        exec_sql(f"""
UPDATE gold.report_runs
SET status='APPROVED',
    approved_by={sql_literal(approved_by)},
    approved_at=current_timestamp()
WHERE report_run_id = '{selected_run_id}';
""")
        st.success(f"Draft promoted to APPROVED: {selected_run_id}")
        st.rerun()