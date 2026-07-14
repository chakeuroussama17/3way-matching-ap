"""
Synthetic historical-invoice generator for demos and model training.

Produces a single tidy CSV whose columns feed *every* model in the pipeline:

============================  ================================================
column                        used by
============================  ================================================
amount                        anomaly, workflow
line_item_count               anomaly, workflow
vendor_invoice_count_30d      anomaly, workflow
days_to_approval              anomaly, workflow (as the approval-time target)
unit_price_deviation_pct      anomaly
po_match_quality              anomaly, workflow
is_duplicate                  anomaly
mismatch_rate                 vendor_risk
payment_delay_days            vendor_risk
compliance_flags              vendor_risk
anomaly_score                 workflow (added post-hoc by the trainer)
risk_score                    workflow (added post-hoc by the trainer)
status / rejected / escalated labels for vendor_risk + workflow
============================  ================================================

The generator is **seeded** so demos are reproducible, and injects a small
fraction of deliberate anomalies (price spikes, duplicates, oversized
invoices) so the unsupervised anomaly model has signal to find.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ml.base import DATA_DIR, get_logger

log = get_logger("data_generator")

# A pool of plausible vendor names; each is assigned an intrinsic reliability
# score at generation time which drives its mismatch / delay / compliance stats.
_VENDOR_POOL = [
    "Acme Corporation", "Globex Industries", "Initech LLC", "Umbrella Supplies",
    "Stark Components", "Wayne Materials", "Wonka Logistics", "Cyberdyne Systems",
    "Soylent Foods", "Hooli Tech", "Pied Piper Ltd", "Vandelay Imports",
    "Gekko Trading", "Oscorp Metals", "Tyrell Manufacturing", "Massive Dynamic",
    "Nakatomi Trading", "Vehement Capital", "Bluth Company", "Prestige Worldwide",
]


def generate_training_data(
    n_rows: int = 500,
    seed: int = 42,
    anomaly_fraction: float = 0.08,
    save: bool = True,
) -> pd.DataFrame:
    """Generate a synthetic historical-invoice dataset.

    Parameters
    ----------
    n_rows
        Number of invoices to generate (100–500 recommended for demos).
    seed
        RNG seed for reproducibility.
    anomaly_fraction
        Approximate share of rows that are deliberately anomalous.
    save
        If ``True`` write the CSV to ``backend/data/sample_training_data.csv``.

    Returns
    -------
    pandas.DataFrame
        The generated dataset, one row per invoice.
    """
    rng = np.random.default_rng(seed)
    n_rows = int(max(50, min(n_rows, 5000)))

    # ── Per-vendor intrinsic profile ─────────────────────────────────────────
    n_vendors = min(len(_VENDOR_POOL), max(6, n_rows // 15))
    vendors = _VENDOR_POOL[:n_vendors]
    # reliability in [0,1]; higher == cleaner vendor
    reliability = rng.beta(5, 2, size=n_vendors)
    vendor_reliability = dict(zip(vendors, reliability))

    base_date = datetime(2026, 1, 1)
    rows = []

    for i in range(n_rows):
        vendor = rng.choice(vendors)
        rel = vendor_reliability[vendor]

        is_anom = rng.random() < anomaly_fraction

        # ── Core invoice features ────────────────────────────────────────────
        line_item_count = int(rng.integers(1, 12))
        base_amount = rng.lognormal(mean=7.5, sigma=0.6)  # ~ $1.8k median
        amount = base_amount * line_item_count / 3
        vendor_invoice_count_30d = int(rng.poisson(4 * rel + 1))

        # deviation of invoice unit price vs PO, in %
        unit_price_deviation_pct = abs(rng.normal(0, (1 - rel) * 3 + 0.5))
        quantity_mismatch = int(rng.random() > (0.85 + 0.1 * rel))
        is_duplicate = int(rng.random() < 0.02)

        # rule-engine match quality (higher for reliable vendors, lower on issues)
        po_match_quality = float(
            np.clip(
                0.7 + 0.3 * rel
                - 0.15 * quantity_mismatch
                - 0.02 * unit_price_deviation_pct
                - 0.2 * is_duplicate
                + rng.normal(0, 0.05),
                0.0,
                1.0,
            )
        )

        # ── Vendor-risk oriented features ────────────────────────────────────
        mismatch_rate = float(np.clip((1 - rel) * 0.6 + rng.normal(0, 0.03), 0, 1))
        payment_delay_days = float(max(0, rng.normal((1 - rel) * 30, 5)))
        compliance_flags = int(rng.poisson((1 - rel) * 2.0))

        # ── Anomaly injection ────────────────────────────────────────────────
        if is_anom:
            kind = rng.choice(["price_spike", "oversized", "duplicate", "off_hours"])
            if kind == "price_spike":
                unit_price_deviation_pct += rng.uniform(15, 40)
                po_match_quality = min(po_match_quality, rng.uniform(0.1, 0.4))
            elif kind == "oversized":
                amount *= rng.uniform(8, 20)
                line_item_count = int(rng.integers(15, 30))
            elif kind == "duplicate":
                is_duplicate = 1
                po_match_quality = min(po_match_quality, 0.35)
            else:  # off_hours == unusually fast/slow approval
                vendor_invoice_count_30d = int(rng.integers(20, 40))

        days_to_approval = float(
            max(
                0.5,
                rng.normal(
                    8
                    + payment_delay_days * 0.1
                    + compliance_flags * 4
                    + (1 - po_match_quality) * 20
                    + quantity_mismatch * 12,
                    4,
                ),
            )
        )

        # ── Outcome label ────────────────────────────────────────────────────
        # Risk of rejection / escalation grows with mismatches and anomalies.
        reject_logit = (
            -3.2
            + 4.5 * mismatch_rate
            + 0.05 * unit_price_deviation_pct
            + 1.5 * quantity_mismatch
            + 2.0 * is_duplicate
            + 2.5 * (1 - po_match_quality)
            + 0.7 * compliance_flags
            + 0.03 * payment_delay_days
        )
        reject_p = 1 / (1 + np.exp(-reject_logit))
        rejected = int(rng.random() < reject_p)

        # Escalation is a *distinct* signal from rejection: high-value invoices
        # require higher approval authority (sign-off), amplified when match
        # quality is weak. It is modelled independently of the rejection gate so
        # the amount signal stays clean for the workflow model (an invoice can be
        # both rejected and escalation-worthy in the labels).
        escalate_logit = (
            -8.0
            + 2.2 * np.log10(amount + 1)
            + 1.0 * (1 - po_match_quality)
        )
        escalate_p = 1 / (1 + np.exp(-escalate_logit))
        escalated = int(rng.random() < escalate_p)

        # Status prioritises rejection for display/audit.
        status = "rejected" if rejected else ("escalated" if escalated else "approved")

        rows.append(
            {
                "invoice_id": f"INV-{2026000 + i}",
                "invoice_date": (base_date + timedelta(days=int(rng.integers(0, 180)))).date().isoformat(),
                "vendor_name": vendor,
                "amount": round(amount, 2),
                "line_item_count": line_item_count,
                "vendor_invoice_count_30d": vendor_invoice_count_30d,
                "days_to_approval": round(days_to_approval, 2),
                "unit_price_deviation_pct": round(unit_price_deviation_pct, 3),
                "quantity_mismatch": quantity_mismatch,
                "po_match_quality": round(po_match_quality, 3),
                "is_duplicate": is_duplicate,
                "mismatch_rate": round(mismatch_rate, 3),
                "payment_delay_days": round(payment_delay_days, 2),
                "compliance_flags": compliance_flags,
                "status": status,
                "rejected": rejected,
                "escalated": escalated,
            }
        )

    df = pd.DataFrame(rows)

    if save:
        out = DATA_DIR / "sample_training_data.csv"
        df.to_csv(out, index=False)
        log.info("Wrote %d synthetic invoices to %s", len(df), out)

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic AP training data.")
    parser.add_argument("--rows", type=int, default=300, help="number of invoices (100-500 typical)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = generate_training_data(n_rows=args.rows, seed=args.seed)
    print(frame.head(10).to_string(index=False))
    print(f"\nGenerated {len(frame)} rows -> backend/data/sample_training_data.csv")
    print("Status distribution:")
    print(frame["status"].value_counts().to_string())
