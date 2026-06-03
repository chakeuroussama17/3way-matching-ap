import streamlit as st
import requests
import pandas as pd

API = "https://threeway-matching-ap.onrender.com"

st.set_page_config(
    page_title="3-Way Matching AP Automation",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ 3-Way Matching AP Automation")
st.caption("Upload PO, GRN, and Invoice → AI extracts fields → Rule engine flags mismatches")

tab1, tab2 = st.tabs(["📤 Submit Documents", "📊 Cases Dashboard"])

# ── TAB 1: Upload & Match ─────────────────────────────────────────────────────
with tab1:
    st.subheader("Upload all 3 documents")
    st.info("Supports PDF, JPG, PNG for all document types.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📋 Purchase Order (PO)**")
        po_file = st.file_uploader(
            "Upload PO",
            type=["pdf", "jpg", "jpeg", "png"],
            key="po"
        )
        if po_file:
            st.success(f"✅ {po_file.name}")

    with col2:
        st.markdown("**📦 Goods Received Note (GRN)**")
        grn_file = st.file_uploader(
            "Upload GRN",
            type=["pdf", "jpg", "jpeg", "png"],
            key="grn"
        )
        if grn_file:
            st.success(f"✅ {grn_file.name}")

    with col3:
        st.markdown("**🧾 Invoice**")
        inv_file = st.file_uploader(
            "Upload Invoice",
            type=["pdf", "jpg", "jpeg", "png"],
            key="inv"
        )
        if inv_file:
            st.success(f"✅ {inv_file.name}")

    all_uploaded = po_file and grn_file and inv_file

    if not all_uploaded:
        st.warning("Please upload all 3 documents to run matching.")

    if st.button("🔍 Run Matching", disabled=not all_uploaded, type="primary"):
        with st.spinner("Extracting data from all 3 documents..."):
            try:
                response = requests.post(
                    f"{API}/match",
                    files={
                        "po":      (po_file.name,  po_file.getvalue()),
                        "grn":     (grn_file.name, grn_file.getvalue()),
                        "invoice": (inv_file.name, inv_file.getvalue())
                    },
                    timeout=120  # GPT calls can take a while
                )
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend. Make sure FastAPI is running on port 8000.")
                st.code("cd backend\nuvicorn main:app --reload --port 8000")
                st.stop()

        # Show backend error clearly instead of cryptic JSONDecodeError
        if response.status_code != 200:
            st.error(f"Backend returned error {response.status_code}")
            st.code(response.text)
            st.stop()

        data = response.json()
        result = data["result"]
        auto = result["auto_approved"]

        st.divider()

        # ── Verdict banner ──
        if auto:
            st.success("✅ AUTO APPROVED — All checks passed. No human review needed.")
        else:
            st.error(f"🚩 FLAGGED FOR REVIEW — {len(result['flags'])} mismatch(es) found.")

        # ── Confidence score ──
        conf = result.get("confidence", 0)
        st.progress(conf, text=f"Match confidence: {int(conf * 100)}%")

        # ── 4 check results ──
        st.subheader("Match checks")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Vendor Match",   "✅ Pass" if result["vendor_match"]   else "❌ Fail")
        c2.metric("Quantity Match", "✅ Pass" if result["quantity_match"] else "❌ Fail")
        c3.metric("Price Match",    "✅ Pass" if result["price_match"]    else "❌ Fail")
        c4.metric("Total Match",    "✅ Pass" if result["total_match"]    else "❌ Fail")

        # ── Diagnosis ──
        if result["flags"]:
            st.subheader("🚩 Mismatches detected")

            # Map each failed check to a plain-English explanation
            diagnoses = {
                "vendor_match": {
                    "title": "Vendor Mismatch",
                    "what": "The vendor name on the Invoice does not match the Purchase Order.",
                    "why":  "This could mean the wrong supplier sent the invoice, or there's a name discrepancy (e.g. abbreviation vs full name).",
                    "action": "Verify the correct supplier and check if a name change was submitted."
                },
                "quantity_match": {
                    "title": "Quantity Mismatch",
                    "what": "The quantity received (GRN) or invoiced does not match what was ordered (PO).",
                    "why":  "Supplier may have short-shipped but is billing for the full ordered amount, or the GRN was recorded incorrectly.",
                    "action": "Cross-check physical delivery records. Only pay for quantity confirmed received in the GRN."
                },
                "price_match": {
                    "title": "Unit Price Mismatch",
                    "what": "The unit price on the Invoice differs from the agreed price on the Purchase Order (beyond 2% tolerance).",
                    "why":  "Price may have changed after PO was issued without a formal amendment, or it's a billing error.",
                    "action": "Contact supplier to issue a corrected invoice or raise a PO amendment if the price change was agreed."
                },
                "total_match": {
                    "title": "Total Amount Mismatch",
                    "what": "The total amount on the Invoice does not match the PO total (beyond 2% tolerance).",
                    "why":  "Could be caused by quantity or price differences, incorrect tax calculation, or unauthorized charges.",
                    "action": "Review line items individually to isolate which item is causing the total discrepancy."
                }
            }

            # Show a diagnosis card for each failed check
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

            # Also show raw flag messages in an expander for technical detail
            with st.expander("🔍 Raw flag details"):
                for flag in result["flags"]:
                    st.code(flag)

        # ── Extracted data side by side ──
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

        # ── Manual approval (only shown if flagged) ──
        if not auto:
            st.divider()
            st.subheader("👤 Manual Review Required")
            st.caption(f"Case ID: {data['case_id']}")

            a1, a2 = st.columns(2)
            with a1:
                if st.button("✅ Manually Approve", type="primary"):
                    r = requests.patch(f"{API}/cases/{data['case_id']}/approve")
                    if r.status_code == 200:
                        st.success("Case manually approved and logged.")
                    else:
                        st.error("Failed to approve. Try again.")

            with a2:
                if st.button("❌ Reject Invoice"):
                    r = requests.patch(f"{API}/cases/{data['case_id']}/reject")
                    if r.status_code == 200:
                        st.error("Case rejected and logged.")
                    else:
                        st.error("Failed to reject. Try again.")

# ── TAB 2: Dashboard ──────────────────────────────────────────────────────────
with tab2:
    st.subheader("All matching cases")

    col_refresh, _ = st.columns([1, 5])
    with col_refresh:
        if st.button("🔄 Refresh"):
            st.rerun()

    try:
        cases_response = requests.get(f"{API}/cases", timeout=10)
        cases = cases_response.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend. Make sure FastAPI is running.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading cases: {e}")
        st.stop()

    if not cases:
        st.info("No cases yet. Submit documents in the first tab.")
    else:
        df = pd.DataFrame(cases)

        # ── Summary metrics ──
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Cases",     len(df))
        m2.metric("Auto Approved",   len(df[df["auto_approved"] == True]))
        m3.metric("Pending Review",  len(df[df["status"] == "pending"]))
        m4.metric("Rejected",        len(df[df["status"] == "rejected"]))

        # ── Styled table ──
        def color_status(val):
            colors = {
                "approved": "background-color: #d4edda; color: #155724",
                "pending":  "background-color: #fff3cd; color: #856404",
                "rejected": "background-color: #f8d7da; color: #721c24"
            }
            return colors.get(val, "")

        styled = df.style.map(color_status, subset=["status"])
        st.dataframe(styled, use_container_width=True)

        # ── Export ──
        st.download_button(
            "⬇️ Export all cases as CSV",
            df.to_csv(index=False),
            file_name="matching_cases.csv",
            mime="text/csv"
        )

        # ── Case drill-down ──
        st.divider()
        st.subheader("🔎 Case detail")
        case_id = st.selectbox("Select case ID to inspect", df["id"].tolist())

        if st.button("Load case detail"):
            try:
                detail = requests.get(f"{API}/cases/{case_id}").json()

                st.markdown(f"**Status:** `{detail['status']}`")

                flags = detail["match_result"].get("flags", [])
                if flags:
                    for flag in flags:
                        st.warning(f"⚠️ {flag}")
                else:
                    st.success("No flags — all checks passed.")

                e1, e2, e3 = st.columns(3)
                with e1:
                    st.markdown("**PO**")
                    st.json(detail["po_data"])
                with e2:
                    st.markdown("**GRN**")
                    st.json(detail["grn_data"])
                with e3:
                    st.markdown("**Invoice**")
                    st.json(detail["invoice_data"])

                # Allow status override from dashboard too
                if detail["status"] == "pending":
                    st.divider()
                    b1, b2 = st.columns(2)
                    with b1:
                        if st.button("✅ Approve from dashboard", key="dash_approve"):
                            requests.patch(f"{API}/cases/{case_id}/approve")
                            st.success("Approved.")
                            st.rerun()
                    with b2:
                        if st.button("❌ Reject from dashboard", key="dash_reject"):
                            requests.patch(f"{API}/cases/{case_id}/reject")
                            st.error("Rejected.")
                            st.rerun()

            except Exception as e:
                st.error(f"Could not load case: {e}")
