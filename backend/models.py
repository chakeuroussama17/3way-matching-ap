from pydantic import BaseModel
from typing import Optional, List

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