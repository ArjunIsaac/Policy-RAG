class ConfidenceEngine:

    def compute(self, chunks, llm_result):

        retrieval_score = self._avg_retrieval(chunks)
        llm_conf = self._llm_confidence(llm_result)
        evidence_score = self._evidence_score(llm_result)

        score = (
            retrieval_score * 0.4 +
            llm_conf * 0.3 +
            evidence_score * 0.3
        )

        return self._label(score)

    # -------------------------
    # RETRIEVAL SCORE
    # -------------------------
    def _avg_retrieval(self, chunks):

        scores = [c.get("score", 0.5) for c in chunks]

        return sum(scores) / len(scores) if scores else 0.0

    # -------------------------
    # LLM CONFIDENCE
    # -------------------------
    def _llm_confidence(self, result):

        conf = result.get("confidence", "low")

        mapping = {
            "low": 0.3,
            "medium": 0.6,
            "high": 0.9
        }

        return mapping.get(conf, 0.3)

    # -------------------------
    # EVIDENCE SCORE
    # -------------------------
    def _evidence_score(self, result):

        evidence = result.get("evidence")

        if not evidence:
            return 0.2

        if len(evidence) > 30:
            return 0.8

        return 0.5

    # -------------------------
    # FINAL LABEL
    # -------------------------
    def _label(self, score):

        if score > 0.75:
            return "high"

        if score > 0.45:
            return "medium"

        return "low"