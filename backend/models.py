from pydantic import BaseModel, Field
from typing import Optional, List, Literal

class LineItem(BaseModel):
    description: Optional[str]
    quantity:    Optional[float]
    unit_price:  Optional[float]
    total:       Optional[float]

class DocumentFields(BaseModel):
    vendor_name:    Optional[str]
    doc_number:     Optional[str]   # PO number, GRN number, or invoice number
    date:           Optional[str]
    line_items:     Optional[List[LineItem]]
    subtotal:       Optional[float]
    tax:            Optional[float]
    total_amount:   Optional[float]

class MatchResult(BaseModel):
    vendor_match:    bool
    quantity_match:  bool
    price_match:     bool
    total_match:     bool
    flags:           List[str]    # human readable mismatch descriptions
    auto_approved:   bool
    confidence:      float        # 0.0 to 1.0


# ── ML request schemas ────────────────────────────────────────────────────────
# All fields have sensible defaults so callers can send partial payloads;
# missing signals fall back to neutral values instead of erroring.

class AnomalyRequest(BaseModel):
    """Features for /detect-anomaly (an invoice's numeric fingerprint)."""
    amount:                   float = Field(0.0,  description="invoice total amount")
    line_item_count:          float = Field(1.0,  description="number of line items")
    vendor_invoice_count_30d: float = Field(1.0,  description="vendor's invoices in last 30 days")
    days_to_approval:         float = Field(5.0,  description="expected/observed days to approval")
    unit_price_deviation_pct: float = Field(0.0,  description="max unit-price deviation vs PO (%)")
    po_match_quality:         float = Field(1.0,  description="rule-engine confidence, 0-1")
    is_duplicate:             float = Field(0.0,  description="1 if a duplicate pattern was detected")
    top_k:                    int   = Field(5,    description="how many features to explain")


class VendorRiskRequest(BaseModel):
    """Features for /vendor-risk."""
    mismatch_rate:       float = Field(0.0, description="historical mismatch rate, 0-1")
    payment_delay_days:  float = Field(0.0, description="average payment delay in days")
    compliance_flags:    float = Field(0.0, description="count of open compliance flags")
    top_k:               int   = Field(5)


class WorkflowRequest(BaseModel):
    """Features for /predict-workflow (approval routing prediction)."""
    amount:                   float = Field(0.0)
    line_item_count:          float = Field(1.0)
    vendor_invoice_count_30d: float = Field(1.0)
    unit_price_deviation_pct: float = Field(0.0)
    quantity_mismatch:        float = Field(0.0, description="1 if quantity check failed")
    is_duplicate:             float = Field(0.0)
    po_match_quality:         float = Field(1.0)
    anomaly_score:            float = Field(0.0, description="0-1 anomaly score (from /detect-anomaly)")
    risk_score:               float = Field(50.0, description="0-100 vendor risk (from /vendor-risk)")
    top_k:                    int   = Field(5)


class ExplainRequest(BaseModel):
    """Explain any model's decision for a given feature vector."""
    model:    Literal["anomaly", "vendor_risk", "workflow"]
    features: dict
    target:   Literal["rejection", "escalation"] = "rejection"  # workflow only
    top_k:    int = 5


class TrainRequest(BaseModel):
    """Trigger a retraining run."""
    rows: int = Field(500, description="synthetic rows to generate if no CSV exists")
    data_path: Optional[str] = Field(None, description="optional path to a historical CSV")