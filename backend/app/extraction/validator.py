import re


class Validator:

    def validate(self, data: dict):

        if not isinstance(data, dict):
            return self._empty("invalid_format")

        return {
            "value": self._normalize_value(data.get("value")),
            "page": self._normalize_page(data.get("page")),
            "clause": self._safe_str(data.get("clause")),
            "evidence": self._safe_str(data.get("evidence")),
            "confidence": self._normalize_confidence(data.get("confidence"))
        }

    # -------------------------
    # VALUE NORMALIZATION
    # -------------------------
    def _normalize_value(self, value):

        if value is None:
            return None

        if isinstance(value, (int, float)):
            return value

        if not isinstance(value, str):
            return str(value)

        text = value.lower().strip()

        # convert written numbers → digits (basic insurance patterns)
        text = self._word_to_number(text)

        return text

    # -------------------------
    # PAGE NORMALIZATION
    # -------------------------
    def _normalize_page(self, page):

        if page is None:
            return None

        try:
            return int(page)
        except:
            return None

    # -------------------------
    # CONFIDENCE NORMALIZATION
    # -------------------------
    def _normalize_confidence(self, conf):

        if not conf:
            return "low"

        conf = str(conf).lower()

        if conf in ["low", "medium", "high"]:
            return conf

        return "low"

    # -------------------------
    # STRING SAFETY
    # -------------------------
    def _safe_str(self, val):

        if val is None:
            return None

        return str(val).strip()

    # -------------------------
    # SIMPLE WORD → NUMBER CONVERTER
    # -------------------------
    def _word_to_number(self, text):

        mappings = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
            "twenty": "20",
            "thirty": "30",
            "forty": "40",
            "fifty": "50",
        }

        for k, v in mappings.items():
            text = re.sub(rf"\b{k}\b", v, text)

        return text

    def _empty(self, reason):
        return {
            "value": None,
            "page": None,
            "clause": None,
            "evidence": None,
            "confidence": "low",
            "error": reason
        }