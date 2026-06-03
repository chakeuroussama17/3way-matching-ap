from typing import Optional

TOLERANCE = 0.02  # 2% tolerance on price/total comparisons

def _safe_float(val) -> Optional[float]:
    try:
        return float(str(val).replace(",", "").replace("$", "").strip())
    except:
        return None

def _items_to_dict(items: list) -> dict:
    """Convert line items list to dict keyed by description for easy comparison"""
    result = {}
    if not items:
        return result
    for item in items:
        desc = str(item.get("description") or "").strip().lower()
        if desc:
            result[desc] = item
    return result

def run_matching(po: dict, grn: dict, invoice: dict) -> dict:
    flags = []
    checks = {}

    # ── 1. Vendor match ──────────────────────────────
    po_vendor  = str(po.get("vendor_name") or "").strip().lower()
    inv_vendor = str(invoice.get("vendor_name") or "").strip().lower()
    checks["vendor_match"] = po_vendor == inv_vendor
    if not checks["vendor_match"]:
        flags.append(f"Vendor mismatch: PO has '{po_vendor}' but invoice has '{inv_vendor}'")

    # ── 2. Quantity match (PO vs GRN vs Invoice) ─────
    po_items  = _items_to_dict(po.get("line_items") or [])
    grn_items = _items_to_dict(grn.get("line_items") or [])
    inv_items = _items_to_dict(invoice.get("line_items") or [])

    quantity_ok = True
    for desc, po_item in po_items.items():
        po_qty  = _safe_float(po_item.get("quantity"))
        grn_qty = _safe_float((grn_items.get(desc) or {}).get("quantity"))
        inv_qty = _safe_float((inv_items.get(desc) or {}).get("quantity"))

        if grn_qty and po_qty and grn_qty < po_qty:
            quantity_ok = False
            flags.append(f"Quantity short for '{desc}': ordered {po_qty}, received {grn_qty}")

        if inv_qty and grn_qty and inv_qty > grn_qty:
            quantity_ok = False
            flags.append(f"Invoice qty exceeds received for '{desc}': invoiced {inv_qty}, received {grn_qty}")

    checks["quantity_match"] = quantity_ok

    # ── 3. Price match (PO vs Invoice) ───────────────
    price_ok = True
    for desc, po_item in po_items.items():
        po_price  = _safe_float(po_item.get("unit_price"))
        inv_price = _safe_float((inv_items.get(desc) or {}).get("unit_price"))

        if po_price and inv_price:
            diff = abs(po_price - inv_price) / po_price
            if diff > TOLERANCE:
                price_ok = False
                flags.append(f"Price mismatch for '{desc}': PO ${po_price}, Invoice ${inv_price}")

    checks["price_match"] = price_ok

    # ── 4. Total match ────────────────────────────────
    po_total  = _safe_float(po.get("total_amount"))
    inv_total = _safe_float(invoice.get("total_amount"))

    if po_total and inv_total:
        diff = abs(po_total - inv_total) / po_total
        checks["total_match"] = diff <= TOLERANCE
        if not checks["total_match"]:
            flags.append(f"Total mismatch: PO ${po_total}, Invoice ${inv_total}")
    else:
        checks["total_match"] = True  # can't check if data missing

    # ── 5. Final decision ─────────────────────────────
    all_pass      = all(checks.values())
    confidence    = round(sum(checks.values()) / len(checks), 2)
    auto_approved = all_pass

    return {
        **checks,
        "flags":        flags,
        "auto_approved": auto_approved,
        "confidence":   confidence
    }