import os

import pandas as pd
import requests
import streamlit as st

# API base resolution (first match wins):
#   1. AP_API_URL env var        — local dev override, e.g. http://localhost:8080
#   2. st.secrets["AP_API_URL"]  — set this on Streamlit Cloud to your Render URL
#   3. http://localhost:8000     — sensible local default
# This keeps local development working out of the box while letting the deployed
# app point at the deployed backend via a secret (no code change needed).
def _resolve_api() -> str:
    if os.getenv("AP_API_URL"):
        return os.getenv("AP_API_URL").rstrip("/")
    try:
        if "AP_API_URL" in st.secrets:
            return str(st.secrets["AP_API_URL"]).rstrip("/")
    except Exception:  # no secrets.toml present — fine, fall through
        pass
    return "http://localhost:8000"


API = _resolve_api()

st.set_page_config(
    page_title="MindHive · AP Intelligence",
    page_icon="⬢",
    layout="wide",
)

# ── MindHive theme — "Nova" style: dark, glassy, violet→cyan gradient accent ──
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{
  --bg-0:#0A0A14; --bg-1:#08080F;
  --surface:rgba(255,255,255,.035); --surface-2:rgba(255,255,255,.06);
  --border:rgba(255,255,255,.09); --border-strong:rgba(255,255,255,.16);
  --ink:#ECECF6; --muted:#9B9BB6; --muted-2:#6E6E88;
  --violet:#7C5CFF; --indigo:#6366F1; --cyan:#22D3EE; --good:#34D399;
  --grad:linear-gradient(90deg,#8B7CFF 0%,#7C5CFF 42%,#22D3EE 100%);
}
html, body, .stApp, [data-testid="stAppViewContainer"]{ font-family:'Inter','Segoe UI',sans-serif; }
.stApp{
  background:
    radial-gradient(720px 460px at 10% -5%, rgba(124,92,255,.22), transparent 60%),
    radial-gradient(620px 520px at 100% 0%, rgba(34,211,238,.07), transparent 55%),
    linear-gradient(180deg,var(--bg-0) 0%,var(--bg-1) 100%);
  background-attachment:fixed;
}
.block-container{ padding-top:1.4rem; max-width:1220px; }

/* ── Hero ── */
.mh-hero{ position:relative; overflow:hidden; border-radius:18px; padding:28px 32px; margin:2px 0 22px 0;
  border:1px solid var(--border);
  background:linear-gradient(180deg,rgba(124,92,255,.10),rgba(255,255,255,.02));
  box-shadow:0 20px 60px rgba(0,0,0,.40); }
.mh-hero::before{ content:""; position:absolute; left:-90px; top:-130px; width:380px; height:380px;
  background:radial-gradient(circle,rgba(124,92,255,.35),transparent 62%); pointer-events:none; }
.mh-hero-inner{ position:relative; display:flex; align-items:flex-start; gap:18px; }
.mh-logo{ width:52px; height:52px; flex:0 0 auto; display:grid; place-items:center; border-radius:14px;
  font-size:25px; color:#fff; background:linear-gradient(135deg,#7C5CFF,#6366F1);
  box-shadow:0 8px 24px rgba(124,92,255,.5); }
.mh-status{ display:inline-flex; align-items:center; gap:7px; font-size:11.5px; font-weight:600;
  color:var(--muted); border:1px solid var(--border); background:rgba(255,255,255,.04);
  padding:4px 11px; border-radius:999px; margin-bottom:12px; }
.mh-status .dot{ width:7px; height:7px; border-radius:50%; background:var(--good); box-shadow:0 0 8px var(--good); }
.mh-title{ font-size:29px; font-weight:800; letter-spacing:-.5px; line-height:1.15; color:var(--ink); }
.mh-title .grad{ background:var(--grad); -webkit-background-clip:text; background-clip:text; color:transparent; }
.mh-sub{ margin-top:8px; font-size:13.5px; color:var(--muted); max-width:640px; line-height:1.55; }
.mh-pills{ margin-top:14px; display:flex; gap:8px; flex-wrap:wrap; }
.mh-pill{ font-size:11px; font-weight:600; color:var(--muted); letter-spacing:.2px;
  border:1px solid var(--border); background:rgba(255,255,255,.03); padding:4px 11px; border-radius:999px; }

/* ── Headings ── */
h1,h2,h3,h4{ color:var(--ink); font-weight:700; letter-spacing:-.2px; }

/* ── Metric cards (glass) ── */
[data-testid="stMetric"]{ background:var(--surface); border:1px solid var(--border);
  border-radius:14px; padding:14px 16px; }
[data-testid="stMetricValue"]{ color:var(--ink); font-weight:800; }
[data-testid="stMetricLabel"]{ color:var(--muted); font-weight:600; }

/* ── Bordered containers → glass cards ── */
[data-testid="stVerticalBlockBorderWrapper"]{ background:var(--surface);
  border:1px solid var(--border)!important; border-radius:16px; }

/* ── Buttons ── */
.stButton>button, .stDownloadButton>button{ border-radius:10px; font-weight:600;
  border:1px solid var(--border); background:var(--surface-2); color:var(--ink); transition:all .15s; }
.stButton>button:hover, .stDownloadButton>button:hover{ border-color:var(--border-strong); background:rgba(255,255,255,.09); }
.stButton>button[kind="primary"]{ background:linear-gradient(90deg,#7C5CFF,#6366F1); color:#fff; border:0; font-weight:600;
  box-shadow:0 8px 22px rgba(124,92,255,.40); }
.stButton>button[kind="primary"]:hover{ filter:brightness(1.08); box-shadow:0 10px 28px rgba(124,92,255,.5); }

/* ── Tabs (gradient underline) ── */
[data-baseweb="tab-list"]{ gap:6px; border-bottom:1px solid var(--border); }
[data-baseweb="tab"]{ font-weight:600; color:var(--muted); background:transparent; padding:8px 14px; }
[data-baseweb="tab"][aria-selected="true"]{ color:var(--ink); }
[data-baseweb="tab-highlight"]{ background:var(--violet); height:3px; border-radius:3px; }

/* inputs / expander / progress / dataframe */
.stNumberInput input, .stTextInput input, [data-baseweb="input"] input{ background:rgba(255,255,255,.04)!important; }
[data-testid="stExpander"]{ border:1px solid var(--border); border-radius:12px; background:var(--surface); }
[data-testid="stProgress"] div[role="progressbar"]>div{ background:linear-gradient(90deg,#7C5CFF,#22D3EE)!important; }
[data-testid="stDataFrame"]{ border:1px solid var(--border); border-radius:12px; }
hr{ border-color:var(--border); }
</style>
"""


def inject_theme():
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def render_hero():
    st.markdown(
        """
        <div class="mh-hero">
          <div class="mh-hero-inner">
            <div class="mh-logo">&#x2B22;</div>
            <div>
              <div class="mh-status"><span class="dot"></span> Rule engine + 5 ML models · live</div>
              <div class="mh-title">Mind<span class="grad">Hive</span> — AP that matches itself,<br>and flags what doesn't.</div>
              <div class="mh-sub">Autonomous 3-way matching for Accounts Payable — extract PO, GRN &amp; Invoice with GPT-4o, detect anomalies, score vendor risk, and route approvals with explainable ML.</div>
              <div class="mh-pills">
                <span class="mh-pill">3-Way Matching</span>
                <span class="mh-pill">Anomaly Detection</span>
                <span class="mh-pill">Vendor Risk</span>
                <span class="mh-pill">Workflow AI</span>
                <span class="mh-pill">Explainable · SHAP</span>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Shared HTTP helpers ───────────────────────────────────────────────────────
def api_get(path: str, **params):
    """GET helper returning (json, error_message)."""
    try:
        r = requests.get(f"{API}{path}", params=params, timeout=30)
        if r.status_code != 200:
            return None, f"{r.status_code}: {r.text}"
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "connection"
    except Exception as e:  # noqa: BLE001
        return None, str(e)


def api_post(path: str, payload: dict, timeout: int = 60):
    """POST helper returning (json, error_message)."""
    try:
        r = requests.post(f"{API}{path}", json=payload, timeout=timeout)
        if r.status_code != 200:
            return None, f"{r.status_code}: {r.text}"
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "connection"
    except Exception as e:  # noqa: BLE001
        return None, str(e)


# ── Reusable ML render components ─────────────────────────────────────────────
_BAND_COLORS = {"red": "🔴", "yellow": "🟡", "green": "🟢", "high": "🔴", "medium": "🟡", "low": "🟢"}


def render_explanation(explanation: dict, caption: str = "Top contributing features"):
    """Render SHAP-style feature contributions as signed bars."""
    if not explanation or not explanation.get("top_features"):
        st.caption("No explanation available.")
        return
    st.caption(f"{caption}  ·  method: `{explanation.get('method', 'n/a')}`")
    feats = explanation["top_features"]
    max_mag = max((abs(f["contribution"]) for f in feats), default=1.0) or 1.0
    for f in feats:
        contrib = f["contribution"]
        arrow = "🔺" if f["direction"] == "increases" else "🔻"
        pct = min(1.0, abs(contrib) / max_mag)
        c1, c2 = st.columns([2, 3])
        with c1:
            st.markdown(f"{arrow} **{f['feature']}** = `{f['value']}`")
        with c2:
            st.progress(pct, text=f"{contrib:+.3f}")


def render_anomaly_card(anomaly: dict):
    """Card for a single invoice's anomaly result."""
    band = anomaly.get("band", "green")
    score = anomaly.get("anomaly_score", 0.0)
    icon = _BAND_COLORS.get(band, "⚪")
    with st.container(border=True):
        c1, c2 = st.columns([1, 2])
        c1.metric("Anomaly score", f"{score:.2f}", help="0 = normal, 1 = highly unusual")
        c2.markdown(f"### {icon} {band.upper()}")
        c2.caption("Flagged as unusual" if anomaly.get("is_anomaly") else "Looks normal")
        if anomaly.get("explanation"):
            with st.expander("❓ Why was this flagged?"):
                render_explanation(anomaly["explanation"], "What made this invoice unusual")


def render_workflow_card(workflow: dict):
    """Card for the workflow prediction + auto-routing recommendation."""
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Rejection prob.", f"{workflow.get('rejection_probability', 0)*100:.0f}%")
        c2.metric("Escalation prob.", f"{workflow.get('escalation_probability', 0)*100:.0f}%")
        c3.metric("Est. approval time", f"{workflow.get('estimated_approval_time_hours', 0):.0f} h")

        route = workflow.get("routing_recommendation", "manual_review")
        if route == "auto_approve":
            st.success("🟢 Recommendation: **AUTO-APPROVE** — low predicted risk.")
        elif route == "manual_review_high_risk":
            st.error("🔴 Recommendation: **MANUAL REVIEW (high risk)** — likely rejection.")
        else:
            st.warning("🟡 Recommendation: **MANUAL REVIEW** — needs a human check.")

        if workflow.get("explanation"):
            with st.expander("❓ Why this prediction?"):
                render_explanation(workflow["explanation"], "Drivers of the rejection prediction")


# ══════════════════════════════════════════════════════════════════════════════
inject_theme()
render_hero()

tab1, tab2, tab3 = st.tabs(["📤 Submit Documents", "📊 Cases Dashboard", "🤖 ML Insights"])

# ── TAB 1: Upload & Match ─────────────────────────────────────────────────────
with tab1:
    st.subheader("Upload all 3 documents")
    st.info("Supports PDF, JPG, PNG for all document types.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📋 Purchase Order (PO)**")
        po_file = st.file_uploader("Upload PO", type=["pdf", "jpg", "jpeg", "png"], key="po")
        if po_file:
            st.success(f"✅ {po_file.name}")
    with col2:
        st.markdown("**📦 Goods Received Note (GRN)**")
        grn_file = st.file_uploader("Upload GRN", type=["pdf", "jpg", "jpeg", "png"], key="grn")
        if grn_file:
            st.success(f"✅ {grn_file.name}")
    with col3:
        st.markdown("**🧾 Invoice**")
        inv_file = st.file_uploader("Upload Invoice", type=["pdf", "jpg", "jpeg", "png"], key="inv")
        if inv_file:
            st.success(f"✅ {inv_file.name}")

    all_uploaded = po_file and grn_file and inv_file
    if not all_uploaded:
        st.warning("Please upload all 3 documents to run matching.")

    if st.button("🔍 Run Matching", disabled=not all_uploaded, type="primary"):
        with st.spinner("Extracting data + running rule engine and ML models..."):
            try:
                response = requests.post(
                    f"{API}/match",
                    files={
                        "po":      (po_file.name,  po_file.getvalue()),
                        "grn":     (grn_file.name, grn_file.getvalue()),
                        "invoice": (inv_file.name, inv_file.getvalue()),
                    },
                    timeout=120,
                )
            except requests.exceptions.ConnectionError:
                st.error(f"❌ Cannot connect to backend at {API}. Make sure FastAPI is running.")
                st.code("cd backend\nuvicorn main:app --reload --port 8000")
                st.stop()

        if response.status_code != 200:
            st.error(f"Backend returned error {response.status_code}")
            st.code(response.text)
            st.stop()

        data = response.json()
        result = data["result"]
        auto = result["auto_approved"]

        st.divider()
        if auto:
            st.success("✅ AUTO APPROVED — All rule checks passed. No human review needed.")
        else:
            st.error(f"🚩 FLAGGED FOR REVIEW — {len(result['flags'])} mismatch(es) found.")

        conf = result.get("confidence", 0)
        st.progress(conf, text=f"Rule-engine match confidence: {int(conf * 100)}%")

        # ── Rule-engine checks ──
        st.subheader("Match checks")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Vendor Match",   "✅ Pass" if result["vendor_match"]   else "❌ Fail")
        c2.metric("Quantity Match", "✅ Pass" if result["quantity_match"] else "❌ Fail")
        c3.metric("Price Match",    "✅ Pass" if result["price_match"]    else "❌ Fail")
        c4.metric("Total Match",    "✅ Pass" if result["total_match"]    else "❌ Fail")

        # ── ML enrichment ──
        ml = result.get("ml") or {}
        if ml.get("available"):
            st.divider()
            st.subheader("🤖 ML insights")
            mcol1, mcol2 = st.columns(2)
            with mcol1:
                st.markdown("**Anomaly detection**")
                if ml.get("anomaly"):
                    render_anomaly_card(ml["anomaly"])
            with mcol2:
                st.markdown("**Approval workflow prediction**")
                if ml.get("workflow"):
                    render_workflow_card(ml["workflow"])
            if ml.get("risk"):
                rs = ml["risk"].get("risk_score")
                src = ml["risk"].get("source")
                st.caption(f"Vendor risk score: **{rs}/100** (source: {src})")
        elif ml:
            st.info("ML enrichment unavailable for this case "
                    f"({ml.get('reason', 'models not trained — run the trainer')}).")

        # ── Diagnosis (existing plain-English cards) ──
        if result["flags"]:
            st.subheader("🚩 Mismatches detected")
            diagnoses = {
                "vendor_match": {
                    "title": "Vendor Mismatch",
                    "what": "The vendor name on the Invoice does not match the Purchase Order.",
                    "why":  "This could mean the wrong supplier sent the invoice, or there's a name discrepancy (e.g. abbreviation vs full name).",
                    "action": "Verify the correct supplier and check if a name change was submitted.",
                },
                "quantity_match": {
                    "title": "Quantity Mismatch",
                    "what": "The quantity received (GRN) or invoiced does not match what was ordered (PO).",
                    "why":  "Supplier may have short-shipped but is billing for the full ordered amount, or the GRN was recorded incorrectly.",
                    "action": "Cross-check physical delivery records. Only pay for quantity confirmed received in the GRN.",
                },
                "price_match": {
                    "title": "Unit Price Mismatch",
                    "what": "The unit price on the Invoice differs from the agreed price on the Purchase Order (beyond 2% tolerance).",
                    "why":  "Price may have changed after PO was issued without a formal amendment, or it's a billing error.",
                    "action": "Contact supplier to issue a corrected invoice or raise a PO amendment if the price change was agreed.",
                },
                "total_match": {
                    "title": "Total Amount Mismatch",
                    "what": "The total amount on the Invoice does not match the PO total (beyond 2% tolerance).",
                    "why":  "Could be caused by quantity or price differences, incorrect tax calculation, or unauthorized charges.",
                    "action": "Review line items individually to isolate which item is causing the total discrepancy.",
                },
            }
            check_keys = {
                "vendor_match":   result.get("vendor_match",   True),
                "quantity_match": result.get("quantity_match", True),
                "price_match":    result.get("price_match",    True),
                "total_match":    result.get("total_match",    True),
            }
            for key, passed in check_keys.items():
                if not passed and key in diagnoses:
                    d = diagnoses[key]
                    with st.container(border=True):
                        st.markdown(f"#### ⚠️ {d['title']}")
                        st.markdown(f"**What happened:** {d['what']}")
                        st.markdown(f"**Likely cause:** {d['why']}")
                        st.markdown(f"**Recommended action:** {d['action']}")
            with st.expander("🔍 Raw flag details"):
                for flag in result["flags"]:
                    st.code(flag)

        # ── Extracted data ──
        st.subheader("Extracted document data")
        d1, d2, d3 = st.columns(3)
        with d1:
            st.markdown("**Purchase Order**")
            st.json(data["po_data"])
        with d2:
            st.markdown("**GRN**")
            st.json(data["grn_data"])
        with d3:
            st.markdown("**Invoice**")
            st.json(data["inv_data"])

        # ── Manual approval ──
        if not auto:
            st.divider()
            st.subheader("👤 Manual Review Required")
            st.caption(f"Case ID: {data['case_id']}")
            a1, a2 = st.columns(2)
            with a1:
                if st.button("✅ Manually Approve", type="primary"):
                    r = requests.patch(f"{API}/cases/{data['case_id']}/approve")
                    st.success("Case manually approved and logged.") if r.status_code == 200 else st.error("Failed to approve.")
            with a2:
                if st.button("❌ Reject Invoice"):
                    r = requests.patch(f"{API}/cases/{data['case_id']}/reject")
                    st.error("Case rejected and logged.") if r.status_code == 200 else st.error("Failed to reject.")

# ── TAB 2: Dashboard ──────────────────────────────────────────────────────────
with tab2:
    st.subheader("All matching cases")
    col_refresh, _ = st.columns([1, 5])
    with col_refresh:
        if st.button("🔄 Refresh"):
            st.rerun()

    cases, err = api_get("/cases")
    if err == "connection":
        st.error(f"❌ Cannot connect to backend at {API}.")
        st.stop()
    elif err:
        st.error(f"Error loading cases: {err}")
        st.stop()

    if not cases:
        st.info("No cases yet. Submit documents in the first tab.")
    else:
        df = pd.DataFrame(cases)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Cases",    len(df))
        m2.metric("Auto Approved",  int((df["auto_approved"] == True).sum()))
        m3.metric("Pending Review", int((df["status"] == "pending").sum()))
        m4.metric("Rejected",       int((df["status"] == "rejected").sum()))
        # % suspicious = red or yellow anomaly bands
        if "anomaly_band" in df.columns:
            susp = df["anomaly_band"].isin(["red", "yellow"]).sum()
            m5.metric("Suspicious (ML)", f"{susp} ({susp/max(len(df),1)*100:.0f}%)")

        def color_status(val):
            colors = {
                "approved": "background-color: #d4edda; color: #155724",
                "pending":  "background-color: #fff3cd; color: #856404",
                "rejected": "background-color: #f8d7da; color: #721c24",
            }
            return colors.get(val, "")

        styled = df.style.map(color_status, subset=["status"])
        st.dataframe(styled, width="stretch")

        st.download_button("⬇️ Export all cases as CSV", df.to_csv(index=False),
                           file_name="matching_cases.csv", mime="text/csv")

        st.divider()
        st.subheader("🔎 Case detail")
        case_id = st.selectbox("Select case ID to inspect", df["id"].tolist())
        if st.button("Load case detail"):
            detail, derr = api_get(f"/cases/{case_id}")
            if derr:
                st.error(f"Could not load case: {derr}")
            else:
                st.markdown(f"**Status:** `{detail['status']}`")
                flags = detail["match_result"].get("flags", [])
                if flags:
                    for flag in flags:
                        st.warning(f"⚠️ {flag}")
                else:
                    st.success("No flags — all checks passed.")

                # Show stored ML block if present
                ml = (detail["match_result"] or {}).get("ml") or {}
                if ml.get("available"):
                    mc1, mc2 = st.columns(2)
                    with mc1:
                        if ml.get("anomaly"):
                            render_anomaly_card(ml["anomaly"])
                    with mc2:
                        if ml.get("workflow"):
                            render_workflow_card(ml["workflow"])

                e1, e2, e3 = st.columns(3)
                with e1:
                    st.markdown("**PO**"); st.json(detail["po_data"])
                with e2:
                    st.markdown("**GRN**"); st.json(detail["grn_data"])
                with e3:
                    st.markdown("**Invoice**"); st.json(detail["invoice_data"])

                if detail["status"] == "pending":
                    st.divider()
                    b1, b2 = st.columns(2)
                    with b1:
                        if st.button("✅ Approve from dashboard", key="dash_approve"):
                            requests.patch(f"{API}/cases/{case_id}/approve")
                            st.success("Approved."); st.rerun()
                    with b2:
                        if st.button("❌ Reject from dashboard", key="dash_reject"):
                            requests.patch(f"{API}/cases/{case_id}/reject")
                            st.error("Rejected."); st.rerun()

# ── TAB 3: ML Insights ────────────────────────────────────────────────────────
with tab3:
    st.subheader("🤖 ML model insights & tools")

    status, serr = api_get("/ml/status")
    if serr == "connection":
        st.error(f"❌ Cannot connect to backend at {API}.")
        st.stop()
    elif serr:
        st.error(f"Error loading ML status: {serr}")
        st.stop()

    # ── Model status strip ──
    trained = {k: v["trained"] for k, v in status["models"].items()}
    caps = status["capabilities"]
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Models trained", f"{sum(trained.values())}/{len(trained)}")
    s2.metric("SHAP explainability", "✅ On" if caps.get("shap") else "⚠️ Fallback")
    s3.metric("Embeddings", "✅ On" if caps.get("embeddings") else "⚠️ Fuzzy")
    with s4:
        if st.button("🔁 Retrain all models"):
            with st.spinner("Retraining on latest data..."):
                res, terr = api_post("/train", {"rows": 500}, timeout=120)
            if terr:
                st.error(f"Training failed: {terr}")
            else:
                st.success("Models retrained.")
                st.json(res["summary"])

    if not all(trained.values()):
        st.warning("Some models are not trained yet. Click **Retrain all models** "
                   "or run `python -m ml.model_trainer` in the backend.")

    st.divider()

    # ── Vendor risk dashboard ──
    st.markdown("### 🏢 Vendor risk dashboard")
    vendors, verr = api_get("/vendors/risk", top=10)
    if verr:
        st.info(f"Vendor risk unavailable: {verr}")
    elif vendors and vendors.get("vendors"):
        vdf = pd.DataFrame(vendors["vendors"])
        show = vdf[["vendor_name", "risk_score", "band", "invoice_count", "incident_count", "incident_rate"]]

        def color_band(val):
            return {"high": "background-color:#f8d7da;color:#721c24",
                    "medium": "background-color:#fff3cd;color:#856404",
                    "low": "background-color:#d4edda;color:#155724"}.get(val, "")

        st.dataframe(show.style.map(color_band, subset=["band"]), width="stretch")
        st.bar_chart(vdf.set_index("vendor_name")["risk_score"], height=280)
    else:
        st.info("No vendor profiles yet — train the models first.")

    st.divider()

    # ── Interactive tools ──
    tcol1, tcol2 = st.columns(2)

    with tcol1:
        st.markdown("### 🔎 Smart match tester")
        st.caption("Hybrid rule + embedding matching with learned aliases.")
        po_text = st.text_input("PO text", "Acme Corporation")
        inv_text = st.text_input("Invoice text", "ACME CORP")
        if st.button("Test match"):
            m, merr = api_get("/smart-match", po_text=po_text, invoice_text=inv_text)
            if merr:
                st.error(merr)
            else:
                st.metric("Confidence", f"{m['confidence']*100:.1f}%")
                st.write("Components:", m["components"])
                st.caption(m["explanation"])
                if m.get("matched_alias"):
                    st.success(f"Learned alias applied: {m['matched_alias']}")

    with tcol2:
        st.markdown("### ⚙️ Anomaly score tester")
        st.caption("Score a hypothetical invoice.")
        amt = st.number_input("Amount", value=52000.0, step=1000.0)
        dev = st.slider("Unit price deviation %", 0.0, 50.0, 28.0)
        poq = st.slider("PO match quality", 0.0, 1.0, 0.3)
        dup = st.checkbox("Duplicate pattern detected", value=True)
        if st.button("Score anomaly"):
            payload = {"amount": amt, "line_item_count": 10, "vendor_invoice_count_30d": 3,
                       "days_to_approval": 8, "unit_price_deviation_pct": dev,
                       "po_match_quality": poq, "is_duplicate": 1 if dup else 0, "top_k": 4}
            a, aerr = api_post("/detect-anomaly", payload)
            if aerr:
                st.error(aerr)
            else:
                render_anomaly_card(a)

    st.divider()

    # ── Monitoring / drift ──
    st.markdown("### 📈 Model monitoring (drift)")
    mon, monerr = api_get("/ml/monitoring")
    if monerr:
        st.info(f"Monitoring unavailable: {monerr}")
    elif mon:
        st.caption(f"Total predictions logged: **{mon['total_predictions_logged']}**")
        rows = []
        for model, d in mon["drift"].items():
            rows.append({
                "model": model,
                "status": d.get("status"),
                "overall_psi": d.get("overall_psi"),
                "severity": d.get("severity"),
                "n_samples": d.get("n_samples"),
                "drift_detected": d.get("drift_detected"),
            })
        st.dataframe(pd.DataFrame(rows), width="stretch")
        st.caption("PSI < 0.10 stable · 0.10–0.25 moderate · > 0.25 major (retrain). "
                   "Small n_samples makes early PSI noisy.")
