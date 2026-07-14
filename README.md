# ⚖️ 3-Way Matching AP Automation

An end-to-end Accounts Payable automation pipeline that ingests Purchase Orders, Goods Received Notes, and Invoices — extracts structured data using GPT-4o Vision — and runs a rule engine that auto-approves clean invoices and flags mismatches for human review.

> Built as part of a portfolio targeting intelligent document processing roles.
> Directly mirrors workflows automated by enterprise AP platforms.

---

## 🎯 The Problem It Solves

In any company that purchases goods, AP teams manually compare 3 documents before approving payment:

- **Purchase Order (PO)** — what was agreed to be bought and at what price
- **Goods Received Note (GRN)** — what was actually delivered
- **Invoice** — what the supplier is billing for

Manually cross-checking these across hundreds of transactions per day is slow, error-prone, and expensive. This system automates the entire process.

---

## 🚀 Live Demo

| Component | URL |
|---|---|
| Frontend (Streamlit) | `https://your-app.streamlit.app` |
| Backend API docs | `https://your-backend.onrender.com/docs` |

---

## ⚙️ How It Works

```
Upload PO + GRN + Invoice (PDF, JPG, PNG)
              ↓
   GPT-4o Vision extracts fields
   from each document independently
              ↓
      Rule engine compares:
      ✓ Vendor name match
      ✓ Quantity: PO vs GRN vs Invoice
      ✓ Unit price: PO vs Invoice
      ✓ Total amount: PO vs Invoice
              ↓
   AUTO APPROVED  ──or──  FLAGGED + diagnosis
              ↓
   Result saved to SQLite database
              ↓
   Dashboard shows full history
```

---

## 🚩 Example: Caught Mismatch

In testing, the system correctly flagged a real-world scenario where:
- PO ordered **50 units** of a cooling fan assembly
- GRN confirmed only **42 units** were received
- Invoice billed for the full **50 units**

**Result:** Quantity mismatch flagged automatically, sent for manual review with a plain-English diagnosis and recommended action — preventing overpayment of 8 units.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Document extraction | GPT-4o Vision (OpenAI) |
| PDF processing | PyMuPDF (no system dependencies) |
| Rule engine | Custom Python logic with 2% price tolerance |
| ML — anomaly detection | scikit-learn IsolationForest |
| ML — vendor risk | scikit-learn LogisticRegression |
| ML — workflow prediction | XGBoost (rejection / escalation / approval-time) |
| ML — explainability | SHAP (TreeExplainer + LinearExplainer) |
| ML — smart matching | sentence-transformers embeddings (fuzzy fallback) + rules |
| ML — monitoring | PSI-based feature-drift detection |
| Backend API | FastAPI + SQLAlchemy |
| Database | SQLite |
| Frontend | Streamlit |
| Backend deployment | Render.com |
| Frontend deployment | Streamlit Cloud |

---

## 📁 Project Structure

```
3way-matching/
├── backend/
│   ├── main.py              # FastAPI endpoints (core + ML)
│   ├── extractor.py         # GPT-4o Vision document parser
│   ├── matcher.py           # Rule engine (vendor, quantity, price, total checks)
│   ├── database.py          # SQLite models + ML-column migration
│   ├── models.py            # Pydantic request/response models
│   ├── ml_integration.py    # Glue: documents/DB ↔ ML models (best-effort)
│   ├── ml/
│   │   ├── base.py                 # persistence, versioning, logging
│   │   ├── data_generator.py       # synthetic historical invoices
│   │   ├── anomaly_detection.py    # IsolationForest
│   │   ├── smart_matching.py       # embeddings + hybrid + alias learning
│   │   ├── explainability.py       # SHAP wrapper (+ fallbacks)
│   │   ├── vendor_risk.py          # LogisticRegression
│   │   ├── workflow_prediction.py  # XGBoost
│   │   ├── monitoring.py           # prediction logging + PSI drift
│   │   ├── model_trainer.py        # unified training entry point
│   │   └── artifacts/              # saved *.joblib models (generated)
│   ├── data/
│   │   └── sample_training_data.csv  # synthetic demo data (generated)
│   ├── Procfile             # Render.com start command
│   └── requirements.txt
├── frontend/
│   ├── app.py               # Streamlit UI (Submit / Dashboard / ML Insights)
│   └── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔌 API Endpoints

**Core pipeline**

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/match` | Upload 3 docs, extract + run rule engine **+ ML enrichment** |
| `GET` | `/cases` | List all historical matching cases (incl. ML scores) |
| `GET` | `/cases/{id}` | Get full detail of a specific case |
| `PATCH` | `/cases/{id}/approve` | Manually approve a flagged case (learns the vendor alias) |
| `PATCH` | `/cases/{id}/reject` | Reject a flagged case |

**ML layer** (see [🤖 Machine Learning Layer](#-machine-learning-layer))

| Method | Endpoint | Description |
|---|---|---|
| `GET`  | `/ml/status` | Which models are trained + active capabilities (SHAP, embeddings) |
| `POST` | `/detect-anomaly` | IsolationForest anomaly score (0-1) + SHAP explanation |
| `POST` | `/vendor-risk` | Logistic-regression vendor risk score (0-100) + explanation |
| `GET`  | `/vendors/risk?top=10` | Top-N highest-risk vendors with incident history |
| `POST` | `/predict-workflow` | XGBoost rejection/escalation probability + auto-routing |
| `GET`  | `/smart-match` | Hybrid rule + embedding match confidence + explanation |
| `POST` | `/explain` | SHAP feature importances for any model decision |
| `GET`  | `/ml/monitoring` | Prediction volume + PSI feature-drift status per model |
| `POST` | `/train` | Retrain all models and hot-reload them |

Full interactive API docs available at `/docs` (FastAPI Swagger UI).

---

## 🤖 Machine Learning Layer

The ML layer **enhances** the rule engine — it never replaces it. The rule engine
remains authoritative; ML adds anomaly scoring, risk, routing predictions and
explanations on top. Every prediction is **explainable** (no black boxes) and the
pipeline **degrades gracefully**: if an optional dependency is missing, the feature
falls back to a simpler always-available implementation instead of crashing.

### The five models

| # | Model | Algorithm | Output |
|---|---|---|---|
| 1 | **Invoice anomaly detection** | IsolationForest | anomaly score 0-1 + red/yellow/green band |
| 2 | **Explainability** | SHAP (Tree/Linear) | top contributing features per decision |
| 3 | **Smart fuzzy matching** | MiniLM embeddings + rules + learned aliases | match confidence 0-1 |
| 4 | **Vendor risk scoring** | LogisticRegression | risk score 0-100 + incident history |
| 5 | **Approval workflow prediction** | XGBoost | rejection / escalation prob + est. time + routing |

The workflow model consumes the anomaly and vendor-risk scores as features, so the
trainer fits models 1 & 4 first and uses them to enrich the data for model 5.

### Train / retrain the models

From the `backend/` directory:

```bash
# First-time setup (also generates synthetic demo data if none exists)
python -m ml.model_trainer

# Regenerate N synthetic invoices, then train
python -m ml.model_trainer --rows 500

# Train on your own historical CSV (see backend/data/sample_training_data.csv
# for the expected columns)
python -m ml.model_trainer --data path/to/history.csv
```

Models are versioned and saved to `backend/ml/artifacts/` (a stable
`<name>.joblib` the API loads on startup, plus an immutable timestamped snapshot
for rollback). The API loads them on startup; you can also retrain live via the
`POST /train` endpoint or the **🔁 Retrain** button in the ML Insights tab.

**Run monthly** on freshly approved data — approvals are logged to
`backend/ml/logs/approved_matches.jsonl` (smart-match aliases) and drift is tracked
in `backend/ml/logs/predictions.jsonl` (PSI vs the training baseline).

### Generate demo data only

```bash
python -m ml.data_generator --rows 300   # writes backend/data/sample_training_data.csv
```

### Example `curl` commands

> Replace `localhost:8000` with your backend URL. All bodies are JSON; every field
> is optional and defaults to a neutral value.

```bash
# 1. Which models are trained + capabilities
curl -s http://localhost:8000/ml/status

# 2. Anomaly detection (returns score + band + SHAP explanation)
curl -s -X POST http://localhost:8000/detect-anomaly \
  -H "Content-Type: application/json" \
  -d '{"amount":52000,"line_item_count":22,"vendor_invoice_count_30d":3,
       "days_to_approval":12,"unit_price_deviation_pct":28,
       "po_match_quality":0.3,"is_duplicate":1,"top_k":3}'

# 3. Vendor risk score for a feature vector
curl -s -X POST http://localhost:8000/vendor-risk \
  -H "Content-Type: application/json" \
  -d '{"mismatch_rate":0.35,"payment_delay_days":28,"compliance_flags":3}'

# 4. Top 10 riskiest vendors
curl -s "http://localhost:8000/vendors/risk?top=10"

# 5. Workflow prediction + auto-routing recommendation
curl -s -X POST http://localhost:8000/predict-workflow \
  -H "Content-Type: application/json" \
  -d '{"amount":52000,"unit_price_deviation_pct":28,"quantity_mismatch":1,
       "is_duplicate":1,"po_match_quality":0.3,"anomaly_score":1.0,
       "risk_score":92.6,"line_item_count":22,"vendor_invoice_count_30d":3}'

# 6. Smart fuzzy match (rule + embedding hybrid)
curl -s "http://localhost:8000/smart-match?po_text=Acme%20Corporation&invoice_text=ACME%20CORP"

# 7. Explain any model decision
curl -s -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{"model":"workflow","target":"escalation",
       "features":{"amount":52000,"po_match_quality":0.3,"anomaly_score":1.0,
                   "risk_score":92.6,"unit_price_deviation_pct":28}}'

# 8. Drift monitoring status
curl -s http://localhost:8000/ml/monitoring

# 9. Retrain all models and hot-reload
curl -s -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" -d '{"rows":500}'
```

### Design constraints honoured

- **Simple, explainable models** — no deep learning in the decision path; SHAP on every prediction.
- **Fast** — single-invoice anomaly + matching complete in well under 500 ms after warm-up.
- **Preserves the rule engine** — ML is additive; `/match` still returns the exact rule result, with ML nested under `result.ml`.
- **Production-ready** — error handling, structured logging, model versioning + snapshots, graceful dependency fallbacks, drift monitoring.

---

## 💻 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/3way-matching-ap.git
cd 3way-matching-ap
```

**2. Set up virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**3. Install dependencies**
```bash
# Full ML stack (XGBoost + SHAP) — recommended for local development:
pip install -r backend/requirements-full.txt
pip install -r frontend/requirements.txt
```

> `backend/requirements.txt` is a **slim** set (scikit-learn only) that installs on
> constrained hosts like Render's free tier. The code auto-detects the heavier
> libraries and falls back gracefully without them: **XGBoost → scikit-learn
> HistGradientBoosting**, **SHAP → coefficient/importance/z-score explanations**,
> **embeddings → rapidfuzz fuzzy matching**. Install `requirements-full.txt` to get
> the full models.

**4. Add your OpenAI key**

Create `backend/.env`:
```
OPENAI_API_KEY=your-key-here
```

**5. Train the ML models** (first run — generates demo data + saves models)
```bash
cd backend
python -m ml.model_trainer
```

**6. Start backend**
```bash
cd backend
uvicorn main:app --reload --port 8000
```

**7. Start frontend** (new terminal)
```bash
cd frontend
streamlit run app.py
```

Open `http://localhost:8501`

> The frontend talks to `http://localhost:8000` by default. Point it elsewhere with
> `AP_API_URL`, e.g. `AP_API_URL=http://localhost:8010 streamlit run app.py`.

---

## 📊 Results

- Processes 3 documents in under 15 seconds end-to-end
- Handles PDF, JPG, and PNG formats without system dependencies
- Correctly flags quantity shortfalls, price discrepancies, and vendor mismatches
- Auto-approves 100% clean invoices with no human intervention required
- Full audit trail stored in database with approve/reject history

---

## 🔗 Related Projects

- [Project 1 — Invoice OCR Parser](https://github.com/YOUR_USERNAME/invoice-ocr-parser)