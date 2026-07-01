import re


class LayoutDetector:

    SECTION_PATTERNS = [

        r"SECTION\s+[A-Z]",

        r"PART\s+[A-Z]",

        r"CLAUSE",

        r"DEFINITIONS",

        r"COVERAGE",

        r"BENEFITS",

        r"WAITING PERIOD",

        r"EXCLUSIONS",

        r"GENERAL EXCLUSIONS",

        r"SPECIFIC EXCLUSIONS",

        r"CLAIMS",

        r"RENEWAL",

        r"PORTABILITY",

        r"GRIEVANCE",

        r"SCHEDULE",

        r"ANNEXURE",

        r"TERMS\s+AND\s+CONDITIONS",

    ]

    def detect_heading(self, line: str):

        clean = line.strip()

        if len(clean) > 120:
            return False

        if clean.isupper():
            return True

        for pattern in self.SECTION_PATTERNS:

            if re.search(pattern, clean, re.IGNORECASE):

                return True

        return False