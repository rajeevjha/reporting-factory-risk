import os, re, json
import pandas as pd
import streamlit as st
from datetime import datetime
from databricks import sql as dbsql
from databricks.sdk.core import Config
from openai import OpenAI

# ── 0) PAGE CONFIG (must be first Streamlit call)
st.set_page_config(page_title="Reporting Factory · Rule Authoring (LLM)", layout="wide")

# ── 1) CONFIG (env or st.secrets)
cfg = Config()
CATALOG        = os.getenv("CATALOG", "reporting_factory_risk_profile")
DATABRICKS_SERVER   = cfg.host or os.getenv("DATABRICKS_HOST")
DATABRICKS_HTTPPATH = os.getenv("DATABRICKS_HTTP_PATH","/sql/1.0/warehouses/5aed87c2fa2d4a3f")
CATALOG             = os.getenv("DBSQL_CATALOG", "reporting_factory_risk_profile")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ── 2) DB helpers (parameterized)
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

# ── 3) Guardrails for condition_sql
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
        issues.append("Unsupported characters. Stick to columns, numbers, quoted strings, AND/OR/NOT, IN, BETWEEN, comparisons, parentheses.")
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

# ── 4) OpenAI (LLM) helpers

try:

    if OPENAI_API_KEY:
        _client = OpenAI(api_key=OPENAI_API_KEY)
        st.caption("✅ OpenAI client initialized.")
    else:
        st.warning("⚠️ OPENAI_API_KEY is missing. Check app.yaml env or secrets scope.")
except ModuleNotFoundError:
    st.error("❌ openai library not found. Run `pip install openai` in your environment.")
except Exception as e:
    st.error(f"🚨 Failed to initialize OpenAI client: {type(e).__name__} — {e}")



SYSTEM_PROMPT = """You generate JSON for risk engine rules.
Return a JSON object with fields:
- rule_id (string)
- name (string)
- segment (string or 'all')
- condition_sql (string, must reference columns as alias f.<col>)
- impact_column ('risk_band'|'risk_points')
- impact_value ('High'|'Medium'|'Low' or integer string like '+20')
- priority (1..100)
- enabled (true/false)
- notes (short description)
Use ONLY columns: f.fico_score, f.dti, f.utilization, f.grade, f.term_months, f.loan_amount, f.interest_rate, f.issue_date, f.annual_income.
Return ONLY JSON (no prose)."""

def llm_suggest_rule(intent: str) -> dict | None:
    if not _client: return None
    msgs = [
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":f"Draft one rule: {intent}\nReturn ONLY JSON."}
    ]
    rsp = _client.chat.completions.create(model=OPENAI_MODEL, messages=msgs, temperature=0.2)
    txt = rsp.choices[0].message.content.strip()
    try:
        start = txt.find("{"); end = txt.rfind("}")
        return json.loads(txt[start:end+1])
    except Exception:
        return None

def llm_explain_change(payload: dict) -> str:
    if not _client: return ""
    msgs = [
        {"role":"system","content":"Summarize the business impact of the change in 1 short paragraph for risk stakeholders."},
        {"role":"user","content":json.dumps(payload, indent=2)}
    ]
    rsp = _client.chat.completions.create(model=OPENAI_MODEL, messages=msgs, temperature=0.3)
    return rsp.choices[0].message.content.strip()

# ── 5) UI – no sidebar, main layout only
st.title("💼 Risk Profiling — Business Rules Editor")

# Top action row
cL, cR = st.columns([2,1], gap="large")
with cL:
    user_intent = st.text_area(
        "Describe the rule you want (natural language):",
        placeholder="e.g., Flag DTI ≥ 45 and Grades E–G as High risk; set high priority.",
        height=120
    )
with cR:
    draft_btn    = st.button("🪄 Draft Rule (LLM)", use_container_width=True)
    validate_btn = st.button("🔍 Validate SQL", use_container_width=True)
    simulate_btn = st.button("▶️ Simulate Impact", use_container_width=True)
    reevaluate_btn = st.button("🔁 Re-Evaluate (Draft)", use_container_width=True)
    save_btn     = st.button("💾 Save / Upsert Rule", use_container_width=True)

st.divider()

# Draft rule editor (stateful)
defaults = {"rule_id":"","name":"","segment":"all","condition_sql":"","impact_column":"risk_band",
            "impact_value":"High","priority":70,"enabled":True,"notes":""}
if "draft_rule" not in st.session_state:
    st.session_state.draft_rule = defaults.copy()

if draft_btn and user_intent.strip():
    cand = llm_suggest_rule(user_intent.strip())
    if cand:
        for k,v in defaults.items(): cand.setdefault(k,v)
        st.session_state.draft_rule = cand
    else:
        st.warning("LLM did not return valid JSON. Refine prompt or check OPENAI_API_KEY.", icon="⚠️")

dr = st.session_state.draft_rule
c1, c2, c3, c4 = st.columns([1,2,2,1.2])
with c1: dr["rule_id"] = st.text_input("rule_id", dr["rule_id"])
with c2: dr["name"]    = st.text_input("name", dr["name"])
with c3: dr["segment"] = st.text_input("segment", dr["segment"])
with c4: dr["priority"]= st.number_input("priority (1–100)", 1, 100, int(dr.get("priority") or 70), 1)

c5, c6, c7 = st.columns([2,1.2,1.2])
with c5:
    dr["condition_sql"] = st.text_area("condition_sql (WHERE expression; alias columns as f.*)",
                                       dr["condition_sql"], height=110)
with c6:
    dr["impact_column"] = st.selectbox("impact_column", ["risk_band","risk_points"],
                                       index=0 if dr.get("impact_column")=="risk_band" else 1)
with c7:
    dr["impact_value"] = st.text_input("impact_value", str(dr["impact_value"]))
c8, c9 = st.columns([1,3])
with c8: dr["enabled"] = st.toggle("enabled", bool(dr.get("enabled", True)))
with c9: dr["notes"]   = st.text_input("notes", dr["notes"])
st.session_state.draft_rule = dr

# ── 6) Validate
if validate_btn:
    problems = validate_rule_payload(dr)
    if problems: st.error("Validation failed:\n- " + "\n- ".join(problems))
    else:        st.success("Validation passed ✅")

# ── 7) Simulate Impact (non-destructive)
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

if simulate_btn:
    probs = validate_rule_payload(dr)
    if probs: st.error("Validation failed:\n- " + "\n- ".join(probs))
    else:
        sim = simulate_impact(dr)
        if "error" in sim: st.error("Simulation error: " + sim["error"])
        else:
            a,b,c,d = st.columns(4)
            a.metric("Total loans", sim["total_loans"])
            b.metric("High (now)", sim["high_now"])
            c.metric("High (after)", sim["high_new"], delta=sim["delta_high"])
            d.metric("% High →", f"{sim['pct_new']}%", delta=f"{round(sim['pct_new']-sim['pct_now'],1)}%")
            st.caption(f"Matches: {sim['matches']} · Newly High: {sim['would_add_to_high']}")
            if _client:
                st.info(llm_explain_change({"rule":dr,"simulation":sim}))

# ── 8) Save / Upsert Rule (parameterized)
def upsert_rule_param(r: dict):
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

if save_btn:
    probs = validate_rule_payload(dr)
    if probs: st.error("Validation failed:\n- " + "\n- ".join(probs))
    else:
        try:
            upsert_rule_param(dr)
            st.success(f"Rule '{dr['rule_id']}' saved ✅")
        except Exception as e:
            st.error(f"Save failed: {e}")

# ── 9) Re-Evaluate (Draft) – rebuild gold.risk_eval from enabled rules (dynamic SQL CTE)
def reevaluate():
    # Pull enabled rules (effective today)
    rules = run_df("""
        SELECT rule_id, impact_column, impact_value, priority, condition_sql
        FROM control.risk_rules
        WHERE enabled = TRUE
          AND current_date BETWEEN effective_from AND COALESCE(effective_to, DATE '2999-12-31')
        ORDER BY priority DESC, rule_id
    """)
    pieces = []
    for _, r in rules.iterrows():
        rule_id = r["rule_id"]
        col = r["impact_column"]
        val = str(r["impact_value"]).replace("'", "''")   # escape single quotes
        pr  = int(r["priority"])
        cond = r["condition_sql"]
        # Basic safety: skip if condition invalid
        ok, issues = validate_condition_sql(cond or "")
        if not ok: continue
        pieces.append(
            f"SELECT f.loan_id, '{rule_id}' AS rule_id, '{col}' AS impact_column, '{val}' AS impact_value, {pr} AS priority "
            f"FROM gold.features f WHERE ({cond})"
        )
    matches_sql = " UNION ALL ".join(pieces) if pieces else "SELECT NULL AS loan_id, NULL AS rule_id, NULL AS impact_column, NULL AS impact_value, 0 AS priority WHERE 1=0"

    stmt = f"""
    CREATE OR REPLACE TABLE gold.risk_eval
    USING DELTA AS
    WITH matches AS (
      {matches_sql}
    ),
    band AS (
      SELECT loan_id,
             FIRST(impact_value) IGNORE NULLS AS risk_band
      FROM (
        SELECT loan_id, impact_value,
               ROW_NUMBER() OVER (PARTITION BY loan_id ORDER BY priority DESC, rule_id) rn
        FROM matches WHERE impact_column='risk_band'
      ) x
      WHERE rn=1
      GROUP BY loan_id
    ),
    points AS (
      SELECT loan_id,
             SUM(TRY_CAST(regexp_extract(impact_value,'[-+]?[0-9]+',0) AS INT)) AS risk_points
      FROM matches WHERE impact_column='risk_points'
      GROUP BY loan_id
    )
    SELECT
      f.*,
      COALESCE(b.risk_band, 'Medium') AS risk_band,
      COALESCE(p.risk_points, 0)      AS risk_points,
      current_timestamp()             AS evaluated_at
    FROM gold.features f
    LEFT JOIN band   b USING(loan_id)
    LEFT JOIN points p USING(loan_id)
    """
    run_exec(stmt)

if reevaluate_btn:
    try:
        with st.spinner("Re-evaluating risk (draft)…"):
            reevaluate()
        st.success("gold.risk_eval rebuilt ✅ (Live KPIs will reflect new classification)")
    except Exception as e:
        st.error(f"Re-Evaluate failed: {e}")

# ── 10) Current rules (quick view)
st.divider()
st.subheader("Current Rules (enabled & effective today)")
rules_df = run_df("""
    SELECT rule_id, name, segment, impact_column, impact_value, priority, enabled, effective_from, owner,
           condition_sql
    FROM control.risk_rules
    WHERE enabled = TRUE
      AND current_date BETWEEN effective_from AND COALESCE(effective_to, DATE '2999-12-31')
    ORDER BY priority DESC, rule_id
""")
st.dataframe(rules_df, use_container_width=True, height=280)