from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import datetime
import json

from database import get_db, MatchCase
from extractor import extract_document
from matcher import run_matching

app = FastAPI(title="3-Way Matching API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/match")
async def match_documents(
    po:      UploadFile = File(...),
    grn:     UploadFile = File(...),
    invoice: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # Extract all 3 documents
    po_data  = extract_document(await po.read(),      po.filename,      "purchase_order")
    grn_data = extract_document(await grn.read(),     grn.filename,     "goods_received_note")
    inv_data = extract_document(await invoice.read(), invoice.filename, "invoice")

    # Run rule engine
    result = run_matching(po_data, grn_data, inv_data)

    # Save to database
    case = MatchCase(
        po_number    = po_data.get("doc_number"),
        vendor       = po_data.get("vendor_name"),
        po_data      = po_data,
        grn_data     = grn_data,
        invoice_data = inv_data,
        match_result = result,
        status       = "approved" if result["auto_approved"] else "pending",
        flags        = len(result["flags"])
    )
    db.add(case)
    db.commit()
    db.refresh(case)

    return {
        "case_id":    case.id,
        "po_data":    po_data,
        "grn_data":   grn_data,
        "inv_data":   inv_data,
        "result":     result
    }


@app.get("/cases")
def get_cases(db: Session = Depends(get_db)):
    cases = db.query(MatchCase).order_by(MatchCase.created_at.desc()).all()
    return [
        {
            "id":           c.id,
            "created_at":   str(c.created_at),
            "po_number":    c.po_number,
            "vendor":       c.vendor,
            "status":       c.status,
            "flags":        c.flags,
            "auto_approved": c.match_result.get("auto_approved"),
            "confidence":   c.match_result.get("confidence")
        }
        for c in cases
    ]


@app.patch("/cases/{case_id}/approve")
def approve_case(case_id: int, db: Session = Depends(get_db)):
    case = db.query(MatchCase).filter(MatchCase.id == case_id).first()
    case.status = "approved"
    db.commit()
    return {"status": "approved"}


@app.patch("/cases/{case_id}/reject")
def reject_case(case_id: int, db: Session = Depends(get_db)):
    case = db.query(MatchCase).filter(MatchCase.id == case_id).first()
    case.status = "rejected"
    db.commit()
    return {"status": "rejected"}


@app.get("/cases/{case_id}")
def get_case_detail(case_id: int, db: Session = Depends(get_db)):
    case = db.query(MatchCase).filter(MatchCase.id == case_id).first()
    return {
        "id":           case.id,
        "po_data":      case.po_data,
        "grn_data":     case.grn_data,
        "invoice_data": case.invoice_data,
        "match_result": case.match_result,
        "status":       case.status
    }