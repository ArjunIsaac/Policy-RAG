from app.llm.ollama import QwenClient
from app.retrieval.retriever import Retriever
from app.extraction.validator import Validator
from app.extraction.confidence import ConfidenceEngine
from app.db.repository import Repository
from app.db.database import Base, engine
from app.db.models import PolicyAttribute
from app.extraction.highlighter import EvidenceHighlighter

#Base.metadata.create_all(bind=engine)


ATTRIBUTE_QUESTIONS = {
    "insurer": "Extract the official insurer company name exactly as written in the document.",
    "policy_name": "Extract the exact product/policy name.",
    "free_look_period": "Extract the Free Look Period duration.",
    "grace_period": "Extract the grace period for premium payment.",
    "ped_waiting_period": "Extract the waiting period for Pre-Existing Diseases (PED).",
    "specific_disease_waiting_period": "Extract waiting period for specific diseases.",
    "initial_waiting_period": "Extract the initial waiting period.",
    "room_rent_limit": "Extract room rent limits or restrictions.",
    "icu_limit": "Extract ICU coverage limits if present.",
    "copay": "Extract co-payment percentage or conditions.",
    "cashless_hospitals": "Extract information about cashless hospital network.",
    "portability": "Extract portability clause details.",
}


class AttributeEngine:

    def __init__(self):

        self.retriever = Retriever()
        self.llm = QwenClient()
        self.validator = Validator()
        self.confidence_engine = ConfidenceEngine()
        self.repo = Repository()
        self.highlighter = EvidenceHighlighter()

    # -------------------------
    # MAIN ENTRY
    # -------------------------
    def extract(self, attribute: str, pages=None):

        if attribute not in ATTRIBUTE_QUESTIONS:
            return {"error": "unknown_attribute"}

        question = ATTRIBUTE_QUESTIONS[attribute]

        # 1. RETRIEVE
        chunks = self.retriever.retrieve(question, top_k=5)

        # 2. CONTEXT
        context = self._build_context(chunks)

        # 3. PROMPT
        prompt = self._build_prompt(attribute, question)

        # 4. LLM CALL
        raw_result = self.llm.generate_json(prompt, context)

        # 5. VALIDATE
        validated = self.validator.validate(raw_result)

        highlight = None

        if pages is not None:
            highlight = self.highlighter.find_evidence(
                pages=pages,
                evidence_text=validated.get("evidence")
            )

        # 6. CONFIDENCE
        confidence = self.confidence_engine.compute(chunks, validated)

        # 7. CONFLICT DETECTION
        conflicts = self._detect_conflicts(chunks, validated)

        # 8. FINAL RESPONSE OBJECT
        final_result = {
            "attribute": attribute,
            "value": validated["value"],
            "page": validated["page"],
            "clause": validated["clause"],
            "evidence": validated["evidence"],
            "confidence": confidence,
            "chunks_used": self._clean_chunks(chunks),
            "conflicts": conflicts,
            "highlight": highlight
        }

        # 9. SAVE TO DATABASE (IMPORTANT FIX)
        self.repo.save_attribute(
            document_id="current_doc",
            attribute=attribute,
            result={
                "value": validated["value"],
                "page": validated["page"],
                "clause": validated["clause"],
                "evidence": validated["evidence"],
                "confidence": confidence,
                "retrieval_score": (
                    sum(c.get("score", 0) * (len(chunks) - i) for i, c in enumerate(chunks))
                    / sum(range(1, len(chunks) + 1))
                    if chunks else None
                ),
                "conflicts": conflicts
            }
        )

        return final_result
    # -------------------------
    # CONTEXT BUILDER
    # -------------------------
    def _build_context(self, chunks):

        blocks = []

        for c in chunks:

            blocks.append(f"""
PAGE: {c['page']}
SECTION: {c['heading']}
SCORE: {c.get('score')}
TEXT: {c['text']}
""")

        return "\n---\n".join(blocks)

    # -------------------------
    # PROMPT BUILDER
    # -------------------------
    def _build_prompt(self, attribute, question):

        return f"""
You are a strict insurance extraction system.

Task: Extract {attribute}

Definition:
{question}

Rules:
- Use ONLY provided context
- Do NOT guess
- If missing, return null
- Return ONLY valid JSON

Required format:
{{
    "value": null,
    "page": null,
    "clause": null,
    "evidence": null,
    "confidence": "low"
}}
"""

    # -------------------------
    # CLEAN CHUNKS
    # -------------------------
    def _clean_chunks(self, chunks):

        return [
            {
                "text": c["text"],
                "page": c["page"],
                "heading": c["heading"],
                "score": c.get("score")
            }
            for c in chunks
        ]

    # -------------------------
    # CONFLICT DETECTION
    # -------------------------
    def _detect_conflicts(self, chunks, result):

        value = result.get("value")

        if not value:
            return []

        conflicts = []

        for c in chunks:

            if value and value.lower() not in c["text"].lower():

                # weak heuristic conflict detection
                if any(keyword in c["text"].lower() for keyword in ["month", "year", "%", "days"]):

                    conflicts.append({
                        "page": c["page"],
                        "text": c["text"][:200]
                    })

        return conflicts