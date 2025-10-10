# ------------------------------------------------------------
# Rule Editor – Simple + Preview (Traditional, non-LLM)
# Databricks SQL only. Minimal UI with preview impact.
# ------------------------------------------------------------

import os
import re
import datetime
import pandas as pd
import streamlit as st
from databricks import sql as dbsql
from databricks.sdk.core import Config

# ────────────────────────────────────────────────────────────
# Page config
# ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Rule Editor – Simple + Preview", layout="wide")
st.title("💼 Risk Profiling — Business Rules Editor (Guided)")
st.caption("Add/manage rules with dropdowns, preview impact, then Re-Evaluate and Approve.")

# ────────────────────────────────────────────────────────────
# Config & DB helpers
# ────────────────────────────────────────────────────────────
cfg = Config()
DATABRICKS_HOST   = cfg.host or os.getenv("DATABRICKS_HOST")
HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH","/sql/1.0/warehouses/5aed87c2fa2d4a3f")
CATALOG             = os.getenv("DBSQL_CATALOG", "reporting_factory_risk_profile")

@st.cache_resource
def get_sql_connection():
    return dbsql.connect(
        server_hostname=DATABRICKS_HOST,
        http_path=HTTP_PATH,
        credentials_provider=lambda: cfg.authenticate,
    )

conn = get_sql_connection()

def run_df(query: str, params: tuple | None = None) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(f"USE CATALOG {CATALOG}")
        cur.execute(query, params or ())
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
    return pd.DataFrame(rows, columns=cols)

def run_exec(query: str, params: tuple | None = None):
    with conn.cursor() as cur:
        cur.execute(f"USE CATALOG {CATALOG}")
        cur.execute(query, params or ())

# ────────────────────────────────────────────────────────────
# Fixed dropdown values (small for demo)
# ────────────────────────────────────────────────────────────
SEGMENTS = ["all", "corporate", "retail"]
IMPACT_COLUMNS = ["risk_band", "risk_points"]
RISK_BANDS = ["High", "Medium", "Low"]

NUMERIC_FIELDS = {
    "fico_score": "FICO score",
    "dti": "Debt-to-Income (%)",
    "utilization": "Revolving Utilization (%)",
    "loan_amount": "Loan Amount",
    "interest_rate": "Interest Rate (%)",
    "annual_income": "Annual Income",
}
CATEGORICAL_FIELDS = {
    "grade": ["A","B","C","D","E","F","G"],
    "term_months": [36, 60],
}
ALL_FIELDS = list(NUMERIC_FIELDS.keys()) + list(CATEGORICAL_FIELDS.keys())

TOKEN_RE = re.compile(r"(?is)^[\s0-9A-Za-z_.'(),=<>\-\+\*/%]+$")

def validate_condition_sql(cond: str) -> list[str]:
    issues = []
    s = (cond or "").strip()
    if not s:
        issues.append("Condition is empty.")
    elif not TOKEN_RE.match(s):
        issues.append("Unsupported characters in condition.")
    cols = set(re.findall(r"\bf\.([A-Za-z_]+)\b", s))
    unknown = [c for c in cols if c not in ALL_FIELDS]
    if unknown:
        issues.append("Unsupported columns: " + ", ".join(sorted(unknown)))
    return issues

def build_condition_sql(field: str, operator: str, values):
    alias = f"f.{field}"
    if field in CATEGORICAL_FIELDS:
        if operator == "=":
            return f"{alias} = '{values}'"
        vals = values if isinstance(values, list) else [values]
        if field == "grade":
            quoted = ",".join([f"'{v}'" for v in vals]) if vals else "''"
            return f"{alias} IN ({quoted})"
        else:
            numbers = ",".join([str(int(v)) for v in vals]) if vals else "0"
            return f"{alias} IN ({numbers})"
    else:
        if operator == "BETWEEN":
            lo, hi = values
            return f"{alias} BETWEEN {float(lo)} AND {float(hi)}"
        else:
            v = float(values)
            return f"{alias} {operator} {v}"

def preview_impact(cond_sql: str, impact_col: str, impact_val: str | int) -> dict:
    """Return dict with matches, avg_dti, avg_fico, high_now, high_after, newly_high, total."""
    # Cohort on features
    stats = run_df(f"""
        SELECT
          COUNT(*) AS matches,
          ROUND(AVG(f.dti), 1) AS avg_dti,
          ROUND(AVG(f.fico_score), 0) AS avg_fico
        FROM gold.features f
        WHERE {cond_sql}
    """)
    matches = int(stats.iloc[0]["matches"] or 0)
    avg_dti = float(stats.iloc[0]["avg_dti"] or 0)
    avg_fico = float(stats.iloc[0]["avg_fico"] or 0)

    # Current high count / total
    cur = run_df("""
        SELECT
          COUNT(*) AS total_loans,
          SUM(CASE WHEN risk_band='High' OR risk_points>=20 THEN 1 ELSE 0 END) AS high_now
        FROM gold.risk_eval
    """)
    total_loans = int(cur.iloc[0]["total_loans"] or 0)
    high_now = int(cur.iloc[0]["high_now"] or 0)

    # Would newly be High due to this rule?
    newly_high = 0
    if impact_col == "risk_band" and str(impact_val).strip().lower() == "high":
        q = f"""
        WITH m AS (SELECT f.loan_id FROM gold.features f WHERE {cond_sql})
        SELECT COUNT(*) AS c
        FROM gold.risk_eval r JOIN m USING(loan_id)
        WHERE NOT (r.risk_band='High' OR r.risk_points>=20)
        """
        newly_high = int(run_df(q).iloc[0]["c"] or 0)
    elif impact_col == "risk_points":
        try:
            add_pts = int(re.findall(r"[-+]?\d+", str(impact_val))[0])
        except Exception:
            add_pts = 0
        q = f"""
        WITH m AS (SELECT f.loan_id FROM gold.features f WHERE {cond_sql})
        SELECT COUNT(*) AS c
        FROM gold.risk_eval r JOIN m USING(loan_id)
        WHERE NOT (r.risk_band='High' OR r.risk_points>=20)
          AND (COALESCE(r.risk_points,0) + {add_pts}) >= 20
        """
        newly_high = int(run_df(q).iloc[0]["c"] or 0)

    high_after = high_now + newly_high
    pct_now = round(100.0 * high_now / total_loans, 1) if total_loans else 0.0
    pct_after = round(100.0 * high_after / total_loans, 1) if total_loans else 0.0

    return dict(
        matches=matches, avg_dti=avg_dti, avg_fico=avg_fico,
        total_loans=total_loans, high_now=high_now,
        high_after=high_after, newly_high=newly_high,
        pct_now=pct_now, pct_after=pct_after
    )

# ────────────────────────────────────────────────────────────
# Add a Rule (with Preview)
# ────────────────────────────────────────────────────────────
st.subheader("➕ Add a Rule")

a1, a2, a3, a4 = st.columns([1.2, 1.5, 1.2, 1.2])
new_rule_id = a1.text_input("Rule ID", value="R_DEMO_1")
new_segment = a2.selectbox("Segment", SEGMENTS, index=0)
impact_col = a3.selectbox("Impact", IMPACT_COLUMNS, index=0)
if impact_col == "risk_band":
    impact_val = a4.selectbox("Value", RISK_BANDS, index=0)
else:
    impact_val = a4.text_input("Points (±int)", value="+15")

b1, b2 = st.columns([1.2, 3])
field = b1.selectbox("Feature", ALL_FIELDS, index=0)
if field in NUMERIC_FIELDS:
    opp = b2.selectbox("Operator", ["<","<=","=",">=",">","BETWEEN"], index=0, key="numop")
    if opp == "BETWEEN":
        c1, c2 = st.columns(2)
        lo = c1.number_input("Low", value=float(600 if field=="fico_score" else 30), step=1.0)
        hi = c2.number_input("High", value=float(700 if field=="fico_score" else 45), step=1.0)
        cond_sql = build_condition_sql(field, "BETWEEN", (lo, hi))
    else:
        val = st.number_input("Value", value=float(680 if field=="fico_score" else 40), step=1.0)
        cond_sql = build_condition_sql(field, opp, val)
else:
    opp = b2.selectbox("Operator", ["=","IN"], index=0, key="catop")
    if field == "grade":
        vals = CATEGORICAL_FIELDS["grade"]
        if opp == "=":
            g = st.selectbox("Grade", vals, index=4)
            cond_sql = build_condition_sql(field, "=", g)
        else:
            gs = st.multiselect("Grades", vals, default=["E","F","G"])
            cond_sql = build_condition_sql(field, "IN", gs)
    else:
        vals = CATEGORICAL_FIELDS["term_months"]
        if opp == "=":
            t = st.selectbox("Term", vals, index=1)
            cond_sql = build_condition_sql(field, "=", t)
        else:
            ts = st.multiselect("Terms", vals, default=[60])
            cond_sql = build_condition_sql(field, "IN", ts)

st.markdown(f"**Condition:** `WHERE {cond_sql}`")

c1, c2, c3 = st.columns([1.2, 1.2, 2])
priority = c1.slider("Priority (1–100)", 1, 100, 80, 1)
enabled = c2.toggle("Enabled", value=True)
notes = c3.text_input("Notes", value="")

# Preview draft rule
pv_col1, pv_col2 = st.columns([1,1])
if pv_col1.button("👁 Preview Draft Impact", use_container_width=True):
    errs = []
    if not re.fullmatch(r"R[A-Za-z0-9_]+", new_rule_id or ""):
        errs.append("Rule ID must start with 'R' and contain letters/numbers/underscore.")
    if impact_col == "risk_points" and not re.fullmatch(r"[+-]?\d+", str(impact_val).strip()):
        errs.append("Points must be integer text like +15 or -10.")
    errs.extend(validate_condition_sql(cond_sql))
    if errs:
        st.error("Fix before preview:\n- " + "\n- ".join(errs))
    else:
        res = preview_impact(cond_sql, impact_col, impact_val)
        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("Matches", res["matches"])
        m2.metric("Avg DTI", res["avg_dti"])
        m3.metric("Avg FICO", res["avg_fico"])
        m4.metric("High (now)", res["high_now"])
        m5.metric("High (after)", res["high_after"], delta=res["high_after"]-res["high_now"])
        st.caption(f"Newly High from this rule: {res['newly_high']}  |  % High: {res['pct_now']}% → {res['pct_after']}%")

# Minimal validation to add
errors = []
if not re.fullmatch(r"R[A-Za-z0-9_]+", new_rule_id or ""):
    errors.append("Rule ID must start with 'R' and contain letters/numbers/underscore.")
if impact_col == "risk_points" and not re.fullmatch(r"[+-]?\d+", str(impact_val).strip()):
    errors.append("Points must be integer text like +15 or -10.")
errors.extend(validate_condition_sql(cond_sql))

if errors:
    st.error("Fix before adding:\n- " + "\n- ".join(errors))

if pv_col2.button("💾 Add Rule", type="primary", disabled=bool(errors), use_container_width=True):
    try:
        eff_from = datetime.date.today().isoformat()
        owner = "ui_simple"
        run_exec("""
            MERGE INTO control.risk_rules AS tgt
            USING (SELECT ? AS rule_id) src
            ON tgt.rule_id = src.rule_id
            WHEN MATCHED THEN UPDATE SET
              name=?,
              segment=?,
              condition_sql=?,
              impact_column=?,
              impact_value=?,
              priority=?,
              enabled=?,
              effective_from=?,
              owner=?,
              notes=?
            WHEN NOT MATCHED THEN INSERT
              (rule_id, name, segment, condition_sql, impact_column, impact_value,
               priority, enabled, effective_from, owner, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            new_rule_id, new_rule_id, new_segment, cond_sql, impact_col, str(impact_val),
            int(priority), bool(enabled), eff_from, owner, notes,
            new_rule_id, new_rule_id, new_segment, cond_sql, impact_col, str(impact_val),
            int(priority), bool(enabled), eff_from, owner, notes
        ))
        st.success(f"Rule {new_rule_id} saved ✅")
    except Exception as e:
        st.error(f"Add failed: {e}")

st.divider()

# ────────────────────────────────────────────────────────────
# Manage a Rule (Enable / Disable / Delete) + Preview existing
# ────────────────────────────────────────────────────────────
st.subheader("🛠 Manage / Preview Existing Rule")

ids_df = run_df("SELECT rule_id FROM control.risk_rules ORDER BY rule_id")
ids = ["(select)"] + list(ids_df["rule_id"]) if not ids_df.empty else ["(select)"]
sel_id = st.selectbox("rule_id", ids, index=0)

mp_col1, mp_col2, mp_col3, mp_col4 = st.columns(4)
if mp_col1.button("👁 Preview Selected", use_container_width=True, disabled=sel_id=="(select)"):
    try:
        rec = run_df("SELECT condition_sql, impact_column, impact_value, priority FROM control.risk_rules WHERE rule_id=?", (sel_id,))
        if rec.empty:
            st.warning("Rule not found.")
        else:
            cond = rec.iloc[0]["condition_sql"]
            ic = rec.iloc[0]["impact_column"]
            iv = rec.iloc[0]["impact_value"]
            bad = validate_condition_sql(cond)
            if bad:
                st.error("Rule has invalid condition:\n- " + "\n- ".join(bad))
            else:
                res = preview_impact(cond, ic, iv)
                m1,m2,m3,m4,m5 = st.columns(5)
                m1.metric("Matches", res["matches"])
                m2.metric("Avg DTI", res["avg_dti"])
                m3.metric("Avg FICO", res["avg_fico"])
                m4.metric("High (now)", res["high_now"])
                m5.metric("High (after)", res["high_after"], delta=res["high_after"]-res["high_now"])
                st.caption(f"Newly High from this rule: {res['newly_high']}  |  % High: {res['pct_now']}% → {res['pct_after']}%")
    except Exception as e:
        st.error(f"Preview failed: {e}")

if mp_col2.button("Enable", use_container_width=True, disabled=sel_id=="(select)"):
    try:
        run_exec("UPDATE control.risk_rules SET enabled=TRUE WHERE rule_id=?", (sel_id,))
        st.success(f"Enabled {sel_id}")
    except Exception as e:
        st.error(f"Enable failed: {e}")

if mp_col3.button("Disable", use_container_width=True, disabled=sel_id=="(select)"):
    try:
        run_exec("UPDATE control.risk_rules SET enabled=FALSE WHERE rule_id=?", (sel_id,))
        st.success(f"Disabled {sel_id}")
    except Exception as e:
        st.error(f"Disable failed: {e}")

confirm = mp_col4.text_input("Type rule_id to delete")
if st.button("🗑 Delete", use_container_width=True, disabled=(sel_id=="(select)")):
    if confirm != sel_id:
        st.error("Confirmation does not match rule_id.")
    else:
        try:
            run_exec("DELETE FROM control.risk_rules WHERE rule_id=?", (sel_id,))
            st.success(f"Deleted {sel_id}")
        except Exception as e:
            st.error(f"Delete failed: {e}")

st.divider()

# ────────────────────────────────────────────────────────────
# Apply rules (Re-Evaluate) and Approve snapshot
# ────────────────────────────────────────────────────────────
st.subheader("⚙️ Apply & Approve")

apply_col, approve_col = st.columns(2)

def reevaluate():
    rules = run_df("""
        SELECT rule_id, impact_column, impact_value, priority, condition_sql
        FROM control.risk_rules
        WHERE enabled = TRUE
          AND current_date BETWEEN effective_from AND COALESCE(effective_to, DATE '2999-12-31')
        ORDER BY priority DESC, rule_id
    """)
    parts = []
    for _, r in rules.iterrows():
        cond = (r["condition_sql"] or "").strip()
        if validate_condition_sql(cond):
            continue
        rule_id_sql = str(r["rule_id"]).replace("'", "''")
        impact_val_sql = str(r["impact_value"]).replace("'", "''")
        parts.append(
            "SELECT f.loan_id, "
            f"'{rule_id_sql}' AS rule_id, "
            f"'{r['impact_column']}' AS impact_column, "
            f"'{impact_val_sql}' AS impact_value, "
            f"{int(r['priority'])} AS priority "
            f"FROM gold.features f WHERE ({cond})"
        )
    matches_sql = " UNION ALL ".join(parts) if parts else "SELECT NULL AS loan_id, NULL AS rule_id, NULL AS impact_column, NULL AS impact_value, 0 AS priority WHERE 1=0"

    stmt = f"""
    CREATE OR REPLACE TABLE gold.risk_eval
    USING DELTA AS
    WITH matches AS ( {matches_sql} ),
    band AS (
      SELECT loan_id, FIRST(impact_value) IGNORE NULLS AS risk_band
      FROM (
        SELECT loan_id, impact_value,
               ROW_NUMBER() OVER (PARTITION BY loan_id ORDER BY priority DESC, rule_id) rn
        FROM matches WHERE impact_column='risk_band'
      ) x WHERE rn=1 GROUP BY loan_id
    ),
    points AS (
      SELECT loan_id,
             SUM(TRY_CAST(regexp_extract(impact_value,'[-+]?[0-9]+',0) AS INT)) AS risk_points
      FROM matches WHERE impact_column='risk_points' GROUP BY loan_id
    )
    SELECT f.*,
           COALESCE(b.risk_band,'Medium') AS risk_band,
           COALESCE(p.risk_points,0)      AS risk_points,
           current_timestamp()            AS evaluated_at
    FROM gold.features f
    LEFT JOIN band b USING(loan_id)
    LEFT JOIN points p USING(loan_id)
    """
    run_exec(stmt)

def approve_snapshot(approved_by="demo_user"):
    run_id = f"RR_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    run_exec("""
        INSERT INTO gold.report_runs
        (report_run_id, name, status, rules_version, started_at, finished_at, approved_by, approved_at)
        VALUES (?, 'Risk Report', 'APPROVED', 'live', current_timestamp(), current_timestamp(), ?, current_timestamp())
    """, (run_id, approved_by))
    kpi = run_df("""
        SELECT COUNT(*) AS loans_total,
               SUM(CASE WHEN risk_band='High' OR risk_points>=20 THEN 1 ELSE 0 END) AS high_risk_count,
               AVG(dti) AS avg_dti, AVG(fico_score) AS avg_fico
        FROM gold.risk_eval
    """).iloc[0]
    run_exec("""
        INSERT INTO gold.report_facts (report_run_id, metric, dimension, value)
        VALUES
          (?, 'loans_total',     'all', ?),
          (?, 'high_risk_count', 'all', ?),
          (?, 'avg_dti',         'all', ?),
          (?, 'avg_fico',        'all', ?)
    """, (run_id, float(kpi["loans_total"] or 0), run_id, float(kpi["high_risk_count"] or 0),
          run_id, float(kpi["avg_dti"] or 0), run_id, float(kpi["avg_fico"] or 0)))
    bands = run_df("SELECT risk_band AS band, COUNT(*) AS loans FROM gold.risk_eval GROUP BY risk_band")
    for _, r in bands.iterrows():
        run_exec("INSERT INTO gold.report_facts (report_run_id, metric, dimension, value) VALUES (?,?,?,?)",
                 (run_id, "count_by_band", r["band"] or "Unknown", float(r["loans"] or 0)))
    return run_id

if apply_col.button("🔁 Re-Evaluate (apply rules)", use_container_width=True):
    try:
        reevaluate()
        st.success("gold.risk_eval rebuilt ✅ (Live metrics updated)")
    except Exception as e:
        st.error(f"Re-Evaluate failed: {e}")

if approve_col.button("✅ Approve snapshot", use_container_width=True):
    try:
        rid = approve_snapshot()
        st.success(f"Approved run created: {rid}  (Approved dashboard will update)")
    except Exception as e:
        st.error(f"Approve failed: {e}")

st.divider()

# ────────────────────────────────────────────────────────────
# Current rules (read-only)
# ────────────────────────────────────────────────────────────
st.subheader("📋 Current Rules (enabled & effective today)")
rules_now = run_df("""
    SELECT rule_id, segment, impact_column, impact_value, priority, enabled, condition_sql, effective_from
    FROM control.risk_rules
    WHERE current_date BETWEEN effective_from AND COALESCE(effective_to, DATE '2999-12-31')
    ORDER BY enabled DESC, priority DESC, rule_id
""")
st.dataframe(rules_now, use_container_width=True, height=320)