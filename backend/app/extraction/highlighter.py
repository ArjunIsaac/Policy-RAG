class EvidenceHighlighter:

    def find_evidence(self, pages, evidence_text):

        if not evidence_text:
            return None

        evidence_lower = evidence_text.lower()

        matches = []

        for page in pages:

            page_text = page["text"].lower()

            if evidence_lower in page_text:

                matches.append({
                    "page": page["page"],
                    "match": page["text"]
                })

        return matches if matches else None