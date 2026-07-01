from app.db.database import SessionLocal
from app.db.models import PolicyAttribute


class Repository:

    def save_attribute(self, document_id: str, attribute: str, result: dict):

        db = SessionLocal()

        record = PolicyAttribute(
            document_id=document_id,
            attribute=attribute,
            value=result.get("value"),
            page=result.get("page"),
            clause=result.get("clause"),
            evidence=result.get("evidence"),
            confidence=result.get("confidence"),
            retrieval_score=result.get("retrieval_score"),
            conflicts=result.get("conflicts")
        )

        db.add(record)
        db.commit()
        db.close()

    def get_document(self, document_id: str):

        db = SessionLocal()

        return db.query(PolicyAttribute)\
            .filter(PolicyAttribute.document_id == document_id)\
            .all()