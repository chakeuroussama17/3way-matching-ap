from sqlalchemy import create_engine, Column, String, Float, JSON, DateTime, Integer, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

engine = create_engine("sqlite:///./matching.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class MatchCase(Base):
    __tablename__ = "cases"

    id            = Column(Integer, primary_key=True, index=True)
    created_at    = Column(DateTime, default=datetime.utcnow)
    po_number     = Column(String)
    vendor        = Column(String)
    po_data       = Column(JSON)   # raw extracted fields
    grn_data      = Column(JSON)
    invoice_data  = Column(JSON)
    match_result  = Column(JSON)   # rule engine output (+ nested "ml" block)
    status        = Column(String, default="pending")  # pending / approved / rejected
    flags         = Column(Integer, default=0)         # number of mismatches

    # ── ML enrichment (scalar columns for fast dashboard querying) ──
    # The full ML detail lives inside match_result["ml"]; these mirror the
    # headline scores so the dashboard can sort/filter without JSON parsing.
    anomaly_score          = Column(Float)   # 0-1, IsolationForest
    anomaly_band           = Column(String)  # red / yellow / green
    risk_score             = Column(Float)   # 0-100, vendor logistic model
    rejection_probability  = Column(Float)   # 0-1, XGBoost workflow model


Base.metadata.create_all(bind=engine)


def _migrate() -> None:
    """Add ML columns to a pre-existing SQLite table (create_all won't ALTER).

    Idempotent: only adds columns that are missing, so it is safe to run on
    every startup regardless of whether the DB predates the ML features.
    """
    new_columns = {
        "anomaly_score": "FLOAT",
        "anomaly_band": "VARCHAR",
        "risk_score": "FLOAT",
        "rejection_probability": "FLOAT",
    }
    with engine.begin() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(cases)"))}
        for name, sql_type in new_columns.items():
            if name not in existing:
                conn.execute(text(f"ALTER TABLE cases ADD COLUMN {name} {sql_type}"))


_migrate()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()