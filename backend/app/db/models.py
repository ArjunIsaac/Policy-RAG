from sqlalchemy import Column, Integer, String, Text, Float, JSON, DateTime
from datetime import datetime

from app.db.database import Base


class PolicyAttribute(Base):

    __tablename__ = "policy_attributes"

    id = Column(Integer, primary_key=True, index=True)

    document_id = Column(String, index=True)

    attribute = Column(String, index=True)

    value = Column(Text)

    page = Column(Integer)

    clause = Column(String)

    evidence = Column(Text)

    confidence = Column(String)

    retrieval_score = Column(Float)

    conflicts = Column(JSON)

    created_at = Column(DateTime, default=datetime.utcnow)