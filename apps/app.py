
import streamlit as st
import databricks.sql as sql
from databricks.sdk.core import Config
import os
import pandas as pd
import textwrap

st.set_page_config(page_title="Risk Rule Builder", layout="wide")
st.title("Risk Rule Builder")



# -------------------------
# Databricks Config & Env Vars
# -------------------------
cfg = Config()
DATABRICKS_HOST = cfg.host or os.getenv("DATABRICKS_HOST")
# Sidebar connection inputs
http_path = st.sidebar.text_input("SQL Warehouse HTTP Path", value=os.getenv("DATABRICKS_HTTP_PATH","/sql/1.0/warehouses/5aed87c2fa2d4a3f"))
catalog = st.sidebar.text_input("Catalog", value=os.getenv("CATALOG_NAME","reporting_factory_risk_profile"))
connect = st.sidebar.button("Connect / Refresh")



# -------------------------
# Databricks SQL Connection
# -------------------------
@st.cache_resource # connection is cached
def get_connection():
    try:
        conn = sql.connect(
        server_hostname=DATABRICKS_HOST,
        http_path=http_path,
        credentials_provider=lambda: cfg.authenticate,
        )
        return conn
    except Exception as e:
        st.error(f"❌ Failed to connect to Databricks SQL: {e}")
        st.stop()

conn = get_connection()
st.success("✅ Connected to Databricks SQL Warehouse")


def run_sql(stmt, params=None, fetch=True):
    if conn is None:
        st.stop()
    with conn.cursor() as c:
        c.execute(f"USE CATALOG {catalog}")
        if params: c.execute(stmt, params)
        else: c.execute(stmt)
        if fetch:
            rows = c.fetchall()
            cols = [d[0] for d in c.description] if c.description else []
            return pd.DataFrame(rows, columns=cols)
        return pd.DataFrame()

# Ensure schema/table exist with one statement per execute
if conn:
    try:
        run_sql("CREATE SCHEMA IF NOT EXISTS control", fetch=False)
        run_sql(textwrap.dedent("""
            CREATE TABLE IF NOT EXISTS control.risk_rules
            (rule_id STRING, name STRING, segment STRING, condition_sql STRING,
             impact_column STRING, impact_value STRING, priority INT, enabled BOOLEAN,
             effective_from DATE, effective_to DATE, owner STRING, notes STRING)
            USING DELTA
        """), fetch=False)
    except Exception as e:
        st.warning(f"Init warning: {e}")

# CRUD UI
if conn:
    st.subheader("Rules (control.risk_rules)")
    rules = run_sql("SELECT * FROM control.risk_rules ORDER BY priority, rule_id")
    st.dataframe(rules, use_container_width=True)

    with st.expander("Add / Update / Delete"):
        tab1, tab2, tab3 = st.tabs(["Add","Update","Delete"])

        with tab1:
            rid = st.text_input("rule_id")
            name = st.text_input("name")
            segment = st.selectbox("segment", ["retail","corporate","all"])
            condition = st.text_area("condition_sql", placeholder="f.fico_score < 620 AND f.dti >= 40")
            impact_col = st.selectbox("impact_column", ["risk_band","risk_points"])
            impact_val = st.text_input("impact_value", value="High")
            priority = st.number_input("priority", 1, 9999, 10, 1)
            enabled = st.checkbox("enabled", True)
            owner = st.text_input("owner", "risk_ops")
            notes = st.text_input("notes", "")
            from datetime import date
            eff_from = st.date_input("effective_from", date.today())
            if st.button("Insert"):
                if not rid or not name or not condition:
                    st.error("rule_id, name, condition_sql are required.")
                else:
                    if impact_col == "risk_points":
                        import re
                        if not re.fullmatch(r"[+\\-]?\\d+", impact_val.strip()):
                            st.error("impact_value must be integer-like for risk_points (e.g., +15)")
                            st.stop()
                    run_sql(textwrap.dedent("""
                        INSERT INTO control.risk_rules
                        (rule_id,name,segment,condition_sql,impact_column,impact_value,priority,enabled,
                         effective_from,effective_to,owner,notes)
                        VALUES (?,?,?,?,?,?,?,?,current_date,NULL,?,?)
                    """), (rid,name,segment,condition,impact_col,impact_val,int(priority),bool(enabled),owner,notes), fetch=False)
                    st.success(f"Inserted {rid}")

        with tab2:
            ridu = st.text_input("rule_id (update)")
            nameu = st.text_input("name (new)")
            segmentu = st.selectbox("segment (new)", ["retail","corporate","all"], key="segU")
            conditionu = st.text_area("condition_sql (new)", key="condU")
            impact_colu = st.selectbox("impact_column (new)", ["risk_band","risk_points"], key="colU")
            impact_valu = st.text_input("impact_value (new)", key="valU")
            priorityu = st.number_input("priority (new)", 1, 9999, 10, 1, key="priU")
            enabledu = st.checkbox("enabled (new)", True, key="enU")
            notesu = st.text_input("notes (new)", key="notesU")
            if st.button("Apply Update"):
                if not ridu:
                    st.error("Provide rule_id.")
                else:
                    if impact_colu == "risk_points":
                        import re
                        if not re.fullmatch(r"[+\\-]?\\d+", impact_valu.strip()):
                            st.error("impact_value must be integer-like for risk_points (e.g., +15)")
                            st.stop()
                    run_sql(textwrap.dedent("""
                        UPDATE control.risk_rules
                        SET name=?, segment=?, condition_sql=?, impact_column=?, impact_value=?, priority=?, enabled=?, notes=?
                        WHERE rule_id=?
                    """), (nameu,segmentu,conditionu,impact_colu,impact_valu,int(priorityu),bool(enabledu),notesu,ridu), fetch=False)
                    st.success(f"Updated {ridu}")

        with tab3:
            ridd = st.text_input("rule_id (delete)")
            if st.button("Delete"):
                if not ridd:
                    st.error("Provide rule_id.")
                else:
                    run_sql("DELETE FROM control.risk_rules WHERE rule_id=?", (ridd,), fetch=False)
                    st.success(f"Deleted {ridd}")

# Preview
if conn:
    st.markdown('---')
    st.subheader("Preview Impact")
    sample_pct = st.slider("Sample percent", 1, 20, 1)
    sample_frac = sample_pct/100.0
    if st.button("Run Preview"):
        rules = run_sql(textwrap.dedent("""
            SELECT rule_id, impact_column, impact_value, priority, condition_sql
            FROM control.risk_rules
            WHERE enabled = TRUE
              AND CURRENT_DATE BETWEEN effective_from AND COALESCE(effective_to, DATE '2999-12-31')
            ORDER BY priority
        """))
        parts = []
        for _, r in rules.iterrows():
            parts.append(
                f"""SELECT loan_id, '{r.rule_id}' AS rule_id, '{r.impact_column}' AS impact_column,
                             '{r.impact_value}' AS impact_value, {int(r.priority)} AS priority
                      FROM {catalog}.gold.features f
                      WHERE rand() < {sample_frac} AND ({r.condition_sql})"""
            )
        union_sql = " UNION ALL ".join(parts) if parts else "SELECT NULL AS loan_id, NULL AS rule_id, NULL AS impact_column, NULL AS impact_value, 0 AS priority WHERE 1=0"
        preview_sql = f"""
            WITH f_s AS (
                SELECT * FROM {catalog}.gold.features WHERE rand() < {sample_frac}
                ),
                matches AS (
                {union_sql}
                ),
                band AS (
                SELECT loan_id, impact_value AS risk_band
                FROM (
                    SELECT loan_id, impact_value, priority,
                        ROW_NUMBER() OVER (PARTITION BY loan_id ORDER BY priority DESC) AS rn
                    FROM matches
                    WHERE impact_column = 'risk_band'
                ) x WHERE rn = 1
                ),
                points AS (
                SELECT loan_id,
                        SUM(CAST(regexp_extract(impact_value, '[-+]?\\d+', 0) AS INT)) AS risk_points
                FROM matches
                WHERE impact_column = 'risk_points'
                GROUP BY loan_id
                ),
                resolved AS (
                SELECT f.*,
                        COALESCE(b.risk_band, 'Medium') AS risk_band,
                        COALESCE(p.risk_points, 0)      AS risk_points
                FROM f_s AS f
                LEFT JOIN band   AS b ON f.loan_id = b.loan_id
                LEFT JOIN points AS p ON f.loan_id = p.loan_id
                )
                SELECT
                COUNT(*)                                        AS loans_total,
                AVG(dti)                                        AS avg_dti,
                AVG(fico_score)                                 AS avg_fico,
                SUM(CASE WHEN risk_band='High' OR risk_points>=20 THEN 1 ELSE 0 END) AS high_risk_count
                FROM resolved
        """
        out = run_sql(preview_sql)
        st.dataframe(out)
        st.success("Preview complete.")

# Build Draft
if conn:
    st.markdown('---')
    st.subheader("Build Draft Report")
    if st.button("Build Draft Now"):
        from datetime import datetime
        run_id = f"RR_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        st.write("Run ID:", run_id)
        rules = run_sql(textwrap.dedent("""
            SELECT rule_id, impact_column, impact_value, priority, condition_sql
            FROM control.risk_rules
            WHERE enabled = TRUE
              AND CURRENT_DATE BETWEEN effective_from AND COALESCE(effective_to, DATE '2999-12-31')
            ORDER BY priority
        """))
        parts = []
        for _, r in rules.iterrows():
            parts.append(
                f"""SELECT loan_id, '{r.rule_id}' AS rule_id, '{r.impact_column}' AS impact_column,
                             '{r.impact_value}' AS impact_value, {int(r.priority)} AS priority
                      FROM {catalog}.gold.features f
                      WHERE ({r.condition_sql})"""
            )
        union_sql = " UNION ALL ".join(parts) if parts else "SELECT NULL AS loan_id, NULL AS rule_id, NULL AS impact_column, NULL AS impact_value, 0 AS priority WHERE 1=0"
        run_sql("""
            INSERT INTO gold.report_runs
            (report_run_id,name,status,rules_version,started_at,finished_at,approved_by,approved_at)
            VALUES (?, 'Risk Report','DRAFT','rules@current',current_timestamp(),NULL,NULL,NULL)
        """, (run_id,), fetch=False)
        kpi_sql = f"""
        WITH matches AS ({union_sql}),
            band AS (
            SELECT loan_id, impact_value AS risk_band
            FROM (
                SELECT loan_id, impact_value, priority,
                    ROW_NUMBER() OVER (PARTITION BY loan_id ORDER BY priority DESC) AS rn
                FROM matches
                WHERE impact_column = 'risk_band'
            ) x WHERE rn = 1
            ),
            points AS (
            SELECT loan_id,
                    SUM(CAST(regexp_extract(impact_value, '[-+]?\\d+', 0) AS INT)) AS risk_points
            FROM matches
            WHERE impact_column = 'risk_points'
            GROUP BY loan_id
            ),
            resolved AS (
            SELECT f.*,
                    COALESCE(b.risk_band, 'Medium') AS risk_band,
                    COALESCE(p.risk_points, 0)      AS risk_points
            FROM {catalog}.gold.features AS f
            LEFT JOIN band   AS b ON f.loan_id = b.loan_id
            LEFT JOIN points AS p ON f.loan_id = p.loan_id
            )
            SELECT * FROM (
            SELECT 'loans_total'   AS metric, 'all' AS dimension, CAST(COUNT(*) AS DOUBLE) AS value FROM resolved
            UNION ALL
            SELECT 'high_risk_count','all', CAST(SUM(CASE WHEN risk_band='High' OR risk_points>=20 THEN 1 ELSE 0 END) AS DOUBLE) FROM resolved
            UNION ALL
            SELECT 'avg_dti','all',  CAST(AVG(dti) AS DOUBLE) FROM resolved
            UNION ALL
            SELECT 'avg_fico','all', CAST(AVG(fico_score) AS DOUBLE) FROM resolved
            )
        """
        kpis = run_sql(kpi_sql)
        for _, row in kpis.iterrows():
            run_sql("""INSERT INTO gold.report_facts (report_run_id,metric,dimension,value) VALUES (?,?,?,?)""",
                    (run_id, row["metric"], row["dimension"], float(row["value"])), fetch=False)
        st.success(f"Draft report created: {run_id}")

# Approve
if conn:
    st.markdown('---')
    st.subheader("Approve Latest Draft")
    if st.button("Approve Most Recent Draft"):
        latest = run_sql("""
            SELECT report_run_id FROM gold.report_runs
            WHERE status='DRAFT' ORDER BY started_at DESC LIMIT 1
        """)
        if latest.empty:
            st.warning("No draft found.")
        else:
            rid = latest.iloc[0,0]
            run_sql("""
                UPDATE gold.report_runs
                SET status='APPROVED', approved_by=current_user(), approved_at=current_timestamp()
                WHERE report_run_id=? AND status='DRAFT'
            """, (rid,), fetch=False)
            run_sql(textwrap.dedent("""
                CREATE OR REPLACE VIEW gold.report_facts_approved_latest AS
                SELECT rf.*
                FROM gold.report_facts rf
                JOIN (
                  SELECT report_run_id
                  FROM gold.report_runs
                  WHERE status='APPROVED'
                  ORDER BY approved_at DESC
                  LIMIT 1
                ) latest USING (report_run_id)
            """), fetch=False)
            st.success(f"Approved run: {rid}")
