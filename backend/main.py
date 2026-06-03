from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
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


@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
        <head>
            <title>3-Way Matching API</title>
        </head>
        <body style="font-family: sans-serif; padding: 40px; background: #0f1117; color: white; max-width: 600px; margin: auto;">
            <h1>⚖️ 3-Way Matching API</h1>
            <p style="color: #aaa;">Accounts Payable automation pipeline — extracts PO, GRN, and Invoice fields using GPT-4o Vision, then runs a rule engine to auto-approve or flag mismatches.</p>
            <hr style="border-color: #333; margin: 24px 0;">
            <h3>🔗 Links</h3>
            <p><a href="/docs" style="color: #4CAF50; text-decoration: none;">📄 Interactive API Docs (Swagger UI)</a></p>
            <p><a href="/redoc" style="color: #4CAF50; text-decoration: none;">📘 ReDoc API Docs</a></p>
            <p><a href="/cases" style="color: #4CAF50; text-decoration: none;">📊 All matching cases (JSON)</a></p>
            <hr style="border-color: #333; margin: 24px 0;">
            <h3>📌 Endpoints</h3>
            <table style="width: 100%; border-collapse: collapse; color: #ccc; font-size: 14px;">
                <tr style="border-bottom: 1px solid #333;">
                    <td style="padding: 8px;"><code style="color: #f0a500;">POST</code></td>
                    <td style="padding: 8px;"><code>/match</code></td>
                    <td style="padding: 8px;">Upload PO + GRN + Invoice, run matching</td>
                </tr>
                <tr style="border-bottom: 1px solid #333;">
                    <td style="padding: 8px;"><code style="color: #4CAF50;">GET</code></td>
                    <td style="padding: 8px;"><code>/cases</code></td>
                    <td style="padding: 8px;">List all historical cases</td>
                </tr>
                <tr style="border-bottom: 1px solid #333;">
                    <td style="padding: 8px;"><code style="color: #4CAF50;">GET</code></td>
                    <td style="padding: 8px;"><code>/cases/{id}</code></td>
                    <td style="padding: 8px;">Get full detail of a case</td>
                </tr>
                <tr style="border-bottom: 1px solid #333;">
                    <td style="padding: 8px;"><code style="color: #2196F3;">PATCH</code></td>
                    <td style="padding: 8px;"><code>/cases/{id}/approve</code></td>
                    <td style="padding: 8px;">Manually approve a flagged case</td>
                </tr>
                <tr>
                    <td style="padding: 8px;"><code style="color: #2196F3;">PATCH</code></td>
                    <td style="padding: 8px;"><code>/cases/{id}/reject</code></td>
                    <td style="padding: 8px;">Reject a flagged case</td>
                </tr>
            </table>
            <hr style="border-color: #333; margin: 24px 0;">
            <p style="color: #555; font-size: 12px;">Built with FastAPI · SQLAlchemy · GPT-4o Vision · PyMuPDF</p>
        </body>
    </html>
    """


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
            "id":            c.id,
            "created_at":    str(c.created_at),
            "po_number":     c.po_number,
            "vendor":        c.vendor,
            "status":        c.status,
            "flags":         c.flags,
            "auto_approved": c.match_result.get("auto_approved"),
            "confidence":    c.match_result.get("confidence")
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
