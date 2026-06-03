from sqlalchemy import create_engine, Column, String, Float, JSON, DateTime, Integer
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
    match_result  = Column(JSON)   # rule engine output
    status        = Column(String, default="pending")  # pending / approved / rejected
    flags         = Column(Integer, default=0)         # number of mismatches

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()