import os

import pandas as pd
import requests
import streamlit as st

# API base is configurable so the frontend isn't locked to one host.
# Defaults to the deployed Render backend; override for local dev with e.g.
#   AP_API_URL=http://localhost:8000  (PowerShell: $env:AP_API_URL="...")
API = os.getenv("AP_API_URL", "https://threeway-matching-ap.onrender.com")

st.set_page_config(
    page_title="MindHive · AP Intelligence",
    page_icon="⬢",
    layout="wide",
)

# ── MindHive theme — restrained slate-navy with a single desaturated gold accent
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root{
  --bg-0:#0E1B2C; --bg-1:#0B1624; --card:#152436; --card-2:#182B41;
  --border:#243851; --border-soft:#1E2E43;
  --ink:#DCE5F0; --muted:#8FA1B8; --muted-2:#61748C;
  --accent:#D8A83E; --accent-l:#E7C170;   /* muted gold, used sparingly */
}
html, body, .stApp, [data-testid="stAppViewContainer"]{ font-family:'Inter','Segoe UI',sans-serif; }
.stApp{ background:linear-gradient(180deg,var(--bg-0) 0%,var(--bg-1) 100%); background-attachment:fixed; }
.block-container{ padding-top:1.3rem; max-width:1220px; }

/* ── Hero (flat, quiet) ── */
.mh-hero{ position:relative; border-radius:12px; padding:22px 26px; margin:2px 0 20px 0;
  border:1px solid var(--border); border-left:3px solid var(--accent);
  background:var(--card); }
.mh-hero-inner{ display:flex; align-items:center; gap:16px; }
.mh-logo{ font-size:34px; line-height:1; color:var(--accent); }
.mh-title{ font-size:25px; font-weight:700; letter-spacing:.2px; color:var(--ink); }
.mh-title span{ color:var(--accent); }
.mh-sub{ margin-top:3px; font-size:13px; color:var(--muted); }
.mh-pills{ margin-top:11px; display:flex; gap:7px; flex-wrap:wrap; }
.mh-pill{ font-size:11px; font-weight:600; color:var(--muted); letter-spacing:.2px;
  border:1px solid var(--border); background:transparent; padding:3px 10px; border-radius:6px; }

/* ── Headings ── */
h1,h2,h3,h4{ color:var(--ink); font-weight:700; letter-spacing:.1px; }

/* ── Metric cards (neutral; value is ink, not accent) ── */
[data-testid="stMetric"]{ background:var(--card); border:1px solid var(--border);
  border-radius:10px; padding:13px 15px; }
[data-testid="stMetricValue"]{ color:var(--ink); font-weight:700; }
[data-testid="stMetricLabel"]{ color:var(--muted); font-weight:600; }

/* ── Bordered containers → flat cards ── */
[data-testid="stVerticalBlockBorderWrapper"]{ background:var(--card);
  border:1px solid var(--border)!important; border-radius:10px; }

/* ── Buttons (accent reserved for primary only) ── */
.stButton>button, .stDownloadButton>button{ border-radius:8px; font-weight:600;
  border:1px solid var(--border); background:var(--card-2); color:var(--ink); transition:all .15s; }
.stButton>button:hover, .stDownloadButton>button:hover{ border-color:var(--muted-2); background:#1D3149; }
.stButton>button[kind="primary"]{ background:var(--accent); color:#20180A; border:1px solid var(--accent);
  font-weight:700; }
.stButton>button[kind="primary"]:hover{ background:var(--accent-l); border-color:var(--accent-l); }

/* ── Tabs (underline accent only) ── */
[data-baseweb="tab-list"]{ gap:4px; border-bottom:1px solid var(--border); }
[data-baseweb="tab"]{ font-weight:600; color:var(--muted); background:transparent; padding:8px 14px; }
[data-baseweb="tab"][aria-selected="true"]{ color:var(--ink); }
[data-baseweb="tab-highlight"]{ background:var(--accent); }

/* inputs / expander / progress / dataframe */
.stNumberInput input, .stTextInput input, [data-baseweb="input"] input{ background:var(--bg-1)!important; }
[data-testid="stExpander"]{ border:1px solid var(--border-soft); border-radius:8px; background:var(--card); }
[data-testid="stProgress"] div[role="progressbar"]>div{ background:var(--accent)!important; }
[data-testid="stDataFrame"]{ border:1px solid var(--border); border-radius:8px; }
hr{ border-color:var(--border-soft); }
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
              <div class="mh-title">Mind<span>Hive</span> &nbsp;·&nbsp; AP Intelligence</div>
              <div class="mh-sub">Autonomous 3-way matching for Accounts Payable — extract, match, and route with ML.</div>
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
