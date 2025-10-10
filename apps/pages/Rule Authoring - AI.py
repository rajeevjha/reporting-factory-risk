# ------------------------------------------------------------
# Reporting Factory · Rule Authoring (LLM) – Guided Flow
# Steps: (1) Draft with AI → (2) Validate & Preview → (3) Save → (4) Re-Evaluate → (5) Approve
# ------------------------------------------------------------

import os, re, json
import pandas as pd
import streamlit as st
from datetime import datetime
from databricks import sql as dbsql
from databricks.sdk.core import Config
from openai import OpenAI

# ────────────────────────────────────────────────────────────
# PAGE CONFIG
# ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Reporting Factory · Rule Authoring", layout="wide")
st.title("💼 Risk Profiling — Business Rules Editor (AI)")
# ────────────────────────────────────────────────────────────
# CONFIG (env vars from Databricks App env)
# ────────────────────────────────────────────────────────────
cfg = Config()
CATALOG        = os.getenv("CATALOG", "reporting_factory_risk_profile")
DATABRICKS_SERVER   = cfg.host or os.getenv("DATABRICKS_HOST")
DATABRICKS_HTTPPATH = os.getenv("DATABRICKS_HTTP_PATH","/sql/1.0/warehouses/5aed87c2fa2d4a3f")
CATALOG             = os.getenv("DBSQL_CATALOG", "reporting_factory_risk_profile")

#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY = ""
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ────────────────────────────────────────────────────────────
# DB helpers (parameterized)
# ────────────────────────────────────────────────────────────
def db_conn():
    return dbsql.connect(
        server_hostname=DATABRICKS_SERVER,
        http_path=DATABRICKS_HTTPPATH,
        credentials_provider=lambda: cfg.authenticate,
    )

def run_df(query: str, params: tuple | None = None) -> pd.DataFrame:
    with db_conn() as c, c.cursor() as cur:
        cur.execute(f"USE CATALOG {CATALOG}")
        cur.execute(query, params or ())
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
    return pd.DataFrame(rows, columns=cols)

def run_exec(query: str, params: tuple | None = None):
    with db_conn() as c, c.cursor() as cur:
        cur.execute(f"USE CATALOG {CATALOG}")
        cur.execute(query, params or ())

# ────────────────────────────────────────────────────────────
# OpenAI client
# ────────────────────────────────────────────────────────────
_client = None
try:
    from openai import OpenAI
    if OPENAI_API_KEY:
        _client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    st.error(f"OpenAI client init failed: {e}")

# ────────────────────────────────────────────────────────────
# Rule validation guardrails
# ────────────────────────────────────────────────────────────
ALLOWED_COLS = {
    "f.fico_score", "f.dti", "f.utilization", "f.grade", "f.term_months",
    "f.loan_amount", "f.interest_rate", "f.issue_date", "f.annual_income"
}
TOKEN_RE = re.compile(r"(?is)^[\s0-9A-Za-z_.'(),=<>\-\+\*/%]+$")

def validate_condition_sql(cond: str) -> tuple[bool, list[str]]:
    issues = []
    s = (cond or "").strip()
    if not s:
        return False, ["condition_sql is empty."]
    if not TOKEN_RE.match(s):
        issues.append("Unsupported characters. Use columns, numbers, quoted strings, AND/OR/NOT, IN, BETWEEN, comparisons, parentheses.")
    if '"' in s:
        issues.append("Use single quotes for string literals. Double quotes are not allowed.")
    used_cols = set(re.findall(r"\bf\.[A-Za-z_]+\b", s))
    unknown = [c for c in used_cols if c not in ALLOWED_COLS]
    if unknown:
        issues.append("Unsupported columns: " + ", ".join(sorted(unknown)))
    if re.search(r"\bgrade\s+IN\s*\(\s*[A-Za-z](\s*,\s*[A-Za-z])+\s*\)", s, flags=re.IGNORECASE):
        issues.append("Quote grade IN-list values, e.g., f.grade IN ('E','F','G').")
    return (len(issues) == 0, issues)

def validate_rule_payload(r: dict) -> list[str]:
    errs = []
    if not r.get("rule_id"): errs.append("rule_id is required.")
    if r.get("impact_column") not in {"risk_band","risk_points"}:
        errs.append("impact_column must be 'risk_band' or 'risk_points'.")
    if r.get("impact_column") == "risk_band" and r.get("impact_value") not in {"High","Medium","Low"}:
        errs.append("For risk_band, impact_value must be 'High' | 'Medium' | 'Low'.")
    if r.get("impact_column") == "risk_points" and not re.fullmatch(r"[+-]?\d+", str(r.get("impact_value","")).strip()):
        errs.append("For risk_points, impact_value must be an integer like '+20' or '-5'.")
    ok, issues = validate_condition_sql(r.get("condition_sql",""))
    if not ok: errs.extend(issues)
    try:
        pr = int(r.get("priority"))
        if pr < 1 or pr > 100: errs.append("priority must be 1..100.")
    except Exception:
        errs.append("priority must be an integer.")
    return errs

# ────────────────────────────────────────────────────────────
# LLM helpers
# ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You generate JSON for risk engine rules.
Return only a JSON object with fields:
rule_id, name, segment, condition_sql, impact_column ('risk_band'|'risk_points'),
impact_value ('High'|'Medium'|'Low' or integer string like '+20'),
priority (1..100), enabled (true/false), notes.
Use ONLY columns alias f.: fico_score, dti, utilization, grade, term_months, loan_amount, interest_rate, issue_date, annual_income.
"""

def llm_suggest_rule(intent: str, debug: bool=False):
    if not _client:
        st.error("OpenAI client not initialized. Check OPENAI_API_KEY and package install.")
        return None
    try:
        rsp = _client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role":"system","content": SYSTEM_PROMPT},
                {"role":"user","content": f"Draft one rule for: {intent}. Return ONLY JSON."}
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
            timeout=45
        )
        raw = rsp.choices[0].message.content
        if debug:
            st.code(raw[:600] + ("…" if len(raw) > 600 else ""), language="json")
        return json.loads(raw)
    except Exception as e:
        st.error(f"LLM request failed: {e}")
        return None

# ────────────────────────────────────────────────────────────
# STATE
# ────────────────────────────────────────────────────────────
defaults = {"rule_id":"","name":"","segment":"all","condition_sql":"",
            "impact_column":"risk_band","impact_value":"High",
            "priority":70,"enabled":True,"notes":""}
if "draft_rule" not in st.session_state:
    st.session_state.draft_rule = defaults.copy()
if "validated_ok" not in st.session_state:
    st.session_state.validated_ok = False
if "preview" not in st.session_state:
    st.session_state.preview = None

# ────────────────────────────────────────────────────────────
# STEP HEADER
# ────────────────────────────────────────────────────────────
st.markdown("### Step 1 → 2 → 3 → 4 → 5")
st.caption("1) Draft with AI  →  2) Validate & Preview  →  3) Save Rule  →  4) Re-Evaluate (apply)  →  5) Approve (snapshot)")

# ────────────────────────────────────────────────────────────
# STEP 1 — DRAFT WITH AI
# ────────────────────────────────────────────────────────────
st.subheader("Step 1 — Draft a rule with AI")
prompt = st.text_area(
    "Describe the rule (natural language)",
    placeholder="E.g., Flag as High when FICO < 680 or utilization > 85; set priority ~90.",
    height=100
)
col1, col2 = st.columns([1,4])
with col1:
    draft_clicked = st.button("🪄 Draft with AI", use_container_width=True, help="Generate rule JSON from your description")

if draft_clicked:
    if not prompt.strip():
        st.warning("Please enter a rule description.")
    else:
        data = llm_suggest_rule(prompt.strip(), debug=False)
        if data:
            for k, v in defaults.items():
                data.setdefault(k, v)
            st.session_state.draft_rule = data
            st.session_state.validated_ok = False
            st.session_state.preview = None
            st.success("Draft created. Proceed to Step 2.")

# Show editable draft
st.markdown("**Draft Rule (editable)**")
dr = st.session_state.draft_rule
c1,c2,c3,c4 = st.columns([1,2,2,1.2])
with c1: dr["rule_id"]   = st.text_input("rule_id", dr["rule_id"])
with c2: dr["name"]      = st.text_input("name", dr["name"])
with c3: dr["segment"]   = st.text_input("segment", dr["segment"])
with c4: dr["priority"]  = st.number_input("priority (1–100)", 1, 100, int(dr.get("priority") or 70), 1)

c5,c6,c7 = st.columns([2,1.2,1.2])
with c5:
    dr["condition_sql"] = st.text_area("condition_sql (use alias f.*)", dr["condition_sql"], height=110)
with c6:
    dr["impact_column"] = st.selectbox("impact_column", ["risk_band","risk_points"],
                                       index=0 if dr.get("impact_column")=="risk_band" else 1)
with c7:
    dr["impact_value"]  = st.text_input("impact_value", str(dr["impact_value"]))
c8,c9 = st.columns([1,3])
with c8: dr["enabled"]  = st.toggle("enabled", bool(dr.get("enabled", True)))
with c9: dr["notes"]    = st.text_input("notes", dr["notes"])
st.session_state.draft_rule = dr

# ────────────────────────────────────────────────────────────
# STEP 2 — VALIDATE & PREVIEW IMPACT
# ────────────────────────────────────────────────────────────
st.subheader("Step 2 — Validate & Preview impact")
vcol1, vcol2 = st.columns([1,1])
validate_clicked = vcol1.button("🔍 Validate rule", use_container_width=True)
preview_clicked  = vcol2.button("👁️ Preview impact (no data change)", use_container_width=True)

def simulate_impact(rule: dict) -> dict:
    cur = run_df("""
        SELECT COUNT(*) AS total_loans,
               SUM(CASE WHEN risk_band='High' OR risk_points>=20 THEN 1 ELSE 0 END) AS high_now
        FROM gold.risk_eval
    """)
    total_loans = int(cur.iloc[0]["total_loans"]); high_now = int(cur.iloc[0]["high_now"])
    ok, issues = validate_condition_sql(rule["condition_sql"])
    if not ok: return {"error":" | ".join(issues)}

    if rule["impact_column"] == "risk_band" and rule["impact_value"] == "High":
        q = f"""
        WITH m AS (SELECT f.loan_id FROM gold.features f WHERE {rule['condition_sql']})
        SELECT (SELECT COUNT(*) FROM m) AS matches,
               (SELECT COUNT(*) FROM gold.risk_eval r JOIN m USING(loan_id)
                 WHERE NOT (r.risk_band='High' OR r.risk_points>=20)) AS would_add
        """
        res = run_df(q); matches = int(res.iloc[0]["matches"]); would_add = int(res.iloc[0]["would_add"])
    elif rule["impact_column"] == "risk_points":
        add_pts = int(str(rule["impact_value"]))
        q = f"""
        WITH m AS (SELECT f.loan_id FROM gold.features f WHERE {rule['condition_sql']})
        SELECT (SELECT COUNT(*) FROM m) AS matches,
               (SELECT COUNT(*) FROM gold.risk_eval r JOIN m USING(loan_id)
                 WHERE NOT (r.risk_band='High' OR r.risk_points>=20)
                   AND (r.risk_points + {add_pts}) >= 20) AS would_add
        """
        res = run_df(q); matches = int(res.iloc[0]["matches"]); would_add = int(res.iloc[0]["would_add"])
    else:
        res = run_df(f"SELECT COUNT(*) AS matches FROM gold.features f WHERE {rule['condition_sql']}")
        matches = int(res.iloc[0]["matches"]); would_add = 0

    high_new = high_now + would_add
    pct_now = round((high_now/total_loans*100.0),1) if total_loans else 0.0
    pct_new = round((high_new/total_loans*100.0),1) if total_loans else 0.0
    return {"total_loans":total_loans,"high_now":high_now,"high_new":high_new,
            "delta_high":high_new-high_now,"matches":matches,"would_add_to_high":would_add,
            "pct_now":pct_now,"pct_new":pct_new}

if validate_clicked:
    problems = validate_rule_payload(dr)
    if problems:
        st.session_state.validated_ok = False
        st.error("Validation failed:\n- " + "\n- ".join(problems))
    else:
        st.session_state.validated_ok = True
        st.success("Validation passed ✅")

if preview_clicked:
    probs = validate_rule_payload(dr)
    if probs:
        st.error("Fix validation issues first:\n- " + "\n- ".join(probs))
    else:
        sim = simulate_impact(dr)
        if "error" in sim:
            st.error("Preview error: " + sim["error"])
        else:
            st.session_state.preview = sim
            a,b,c,d = st.columns(4)
            a.metric("Total loans", sim["total_loans"])
            b.metric("High (now)", sim["high_now"])
            c.metric("High (after)", sim["high_new"], delta=sim["delta_high"])
            d.metric("% High →", f"{sim['pct_new']}%", delta=f"{round(sim['pct_new']-sim['pct_now'],1)}%")
            st.caption(f"Matches: {sim['matches']} · Newly High: {sim['would_add_to_high']}")

# ────────────────────────────────────────────────────────────
# STEP 3 — SAVE RULE
# ────────────────────────────────────────────────────────────
st.subheader("Step 3 — Save rule to catalog (control.risk_rules)")
save_disabled = not st.session_state.validated_ok
save_clicked = st.button("💾 Save / Upsert rule", disabled=save_disabled, use_container_width=True)

def upsert_rule(r: dict):
    priority = int(r["priority"])
    enabled = bool(r["enabled"])
    effective_from = datetime.utcnow().date().isoformat()
    # UPDATE
    run_exec("""
        UPDATE control.risk_rules
        SET name=?, segment=?, condition_sql=?, impact_column=?, impact_value=?,
            priority=?, enabled=?, effective_from=?, effective_to=NULL, owner=?, notes=?
        WHERE rule_id=?
    """, (r["name"] or None, r["segment"] or "all", r["condition_sql"], r["impact_column"],
          str(r["impact_value"]), priority, enabled, effective_from, "ui_demo", r["notes"] or None, r["rule_id"]))
    # INSERT if not exists
    exists = run_df("SELECT COUNT(*) AS c FROM control.risk_rules WHERE rule_id=?", (r["rule_id"],)).iloc[0]["c"] > 0
    if not exists:
        run_exec("""
            INSERT INTO control.risk_rules
            (rule_id, name, segment, condition_sql, impact_column, impact_value,
             priority, enabled, effective_from, effective_to, owner, notes)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (r["rule_id"], r["name"] or None, r["segment"] or "all", r["condition_sql"],
              r["impact_column"], str(r["impact_value"]), priority, enabled, effective_from, None, "ui_demo", r["notes"] or None))

if save_clicked:
    try:
        upsert_rule(dr)
        st.success(f"Rule '{dr['rule_id']}' saved to control.risk_rules ✅")
    except Exception as e:
        st.error(f"Save failed: {e}")

# ────────────────────────────────────────────────────────────
# STEP 4 — RE-EVALUATE (APPLY RULES)
# ────────────────────────────────────────────────────────────
st.subheader("Step 4 — Re-Evaluate (apply enabled rules → gold.risk_eval)")
reeval_clicked = st.button("🔁 Re-Evaluate (rebuild gold.risk_eval)", use_container_width=True)

def reevaluate():
    rules = run_df("""
        SELECT rule_id, impact_column, impact_value, priority, condition_sql
        FROM control.risk_rules
        WHERE enabled = TRUE
          AND current_date BETWEEN effective_from AND COALESCE(effective_to, DATE '2999-12-31')
        ORDER BY priority DESC, rule_id
    """)
    pieces = []
    for _, r in rules.iterrows():
        ok, _ = validate_condition_sql(r["condition_sql"] or "")
        if not ok: 
            continue
        rule_id_sql = r["rule_id"].replace("'", "''")
        impact_val_sql = str(r["impact_value"]).replace("'", "''")

        pieces.append(
            "SELECT f.loan_id, "
            f"'{rule_id_sql}' AS rule_id, "
            f"'{r['impact_column']}' AS impact_column, "
            f"'{impact_val_sql}' AS impact_value, "
            f"{int(r['priority'])} AS priority "
            f"FROM gold.features f WHERE ({r['condition_sql']})"
        )
    matches_sql = " UNION ALL ".join(pieces) if pieces else "SELECT NULL AS loan_id, NULL AS rule_id, NULL AS impact_column, NULL AS impact_value, 0 AS priority WHERE 1=0"

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

if reeval_clicked:
    try:
        with st.spinner("Re-evaluating…"):
            reevaluate()
        st.success("gold.risk_eval rebuilt ✅ (Live tiles will reflect new classification)")
    except Exception as e:
        st.error(f"Re-Evaluate failed: {e}")

# ────────────────────────────────────────────────────────────
# STEP 5 — APPROVE (SNAPSHOT KPIs)
# ────────────────────────────────────────────────────────────
st.subheader("Step 5 — Approve (snapshot KPIs → approved dashboard)")
approve_clicked = st.button("✅ Approve snapshot", use_container_width=True)

def approve_snapshot(approved_by="demo_user"):
    run_id = f"RR_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
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
    """, (run_id, float(kpi["loans_total"]), run_id, float(kpi["high_risk_count"]),
          run_id, float(kpi["avg_dti"] or 0), run_id, float(kpi["avg_fico"] or 0)))
    bands = run_df("SELECT risk_band AS band, COUNT(*) AS loans FROM gold.risk_eval GROUP BY risk_band")
    for _, r in bands.iterrows():
        run_exec("INSERT INTO gold.report_facts (report_run_id, metric, dimension, value) VALUES (?,?,?,?)",
                 (run_id, "count_by_band", r["band"] or "Unknown", float(r["loans"])))
    return run_id

if approve_clicked:
    try:
        rid = approve_snapshot()
        st.success(f"Approved run created: {rid} ✅  (Approved tiles will update)")
    except Exception as e:
        st.error(f"Approve failed: {e}")

# ────────────────────────────────────────────────────────────
# CURRENT RULES VIEW
# ────────────────────────────────────────────────────────────
st.markdown("### Enabled rules (effective today)")
rules_df = run_df("""
    SELECT rule_id, name, segment, impact_column, impact_value, priority, enabled, effective_from, owner, condition_sql
    FROM control.risk_rules
    WHERE enabled = TRUE
      AND current_date BETWEEN effective_from AND COALESCE(effective_to, DATE '2999-12-31')
    ORDER BY priority DESC, rule_id
""")
st.dataframe(rules_df, use_container_width=True, height=260)