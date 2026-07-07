"""
regex_extractor.py - Extract simple attributes with regex (0 LLM calls)
Handles conditions where possible, flags for LLM otherwise.
"""

import re
from typing import Dict, Any, List, Optional, Tuple


class RegexExtractor:
    """Extract simple attributes using regex patterns."""
    
    @staticmethod
    def extract_all(text: str) -> Dict[str, Dict]:
        """Extract all regex-able attributes."""
        results = {}
        
        # Simple fields (no complex conditions)
        simple_extractors = [
            ("sum_insured_options", RegexExtractor._extract_sum_insured),
            ("copay_percentage", RegexExtractor._extract_copay),
            ("grace_period_days", RegexExtractor._extract_grace_period),
            ("free_look_period_days", RegexExtractor._extract_free_look),
            ("claim_settlement_days", RegexExtractor._extract_claim_settlement),
        ]
        
        for field_name, extractor in simple_extractors:
            result = extractor(text)
            if result:
                results[field_name] = result
        
        # Complex fields with conditions - extract what we can, flag for LLM
        conditional_results = RegexExtractor._extract_conditional_fields(text)
        results.update(conditional_results)
        
        return results
    
    @staticmethod
    def _extract_sum_insured(text: str) -> Optional[Dict]:
        """Extract sum insured options."""
        patterns = [
            r'Sum\s+Insured\s*(?:options?)?\s*[:\-]\s*([^\n\.]+)',
            r'[Ss]um\s+[Aa]ssured\s*(?:options?)?\s*[:\-]\s*([^\n\.]+)',
            r'[Cc]overage\s*(?:options?)?\s*[:\-]\s*([^\n\.]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value_str = match.group(1).strip()
                # Check if there's a condition (e.g., "subject to", "excluding")
                condition_match = re.search(r'(?:subject to|excluding|except)([^,;]+)', value_str)
                if condition_match:
                    # Has condition - store both value and condition
                    values = re.split(r'[,;|]', value_str.split('subject')[0].strip())
                    values = [v.strip() for v in values if v.strip()]
                    return {
                        "value": values,
                        "display": ", ".join(values),
                        "page": None,
                        "confidence": 0.85,  # Slightly lower due to condition
                        "status": "requires_verification",
                        "evidence": match.group(0)[:200],
                        "reasoning": f"Extracted via regex with condition: {condition_match.group(1)[:100]}",
                        "_condition": condition_match.group(1)  # Store for LLM verification
                    }
                else:
                    # No condition - clean extraction
                    values = re.split(r'[,;|]', value_str)
                    values = [v.strip() for v in values if v.strip()]
                    if values:
                        return {
                            "value": values,
                            "display": ", ".join(values),
                            "page": None,
                            "confidence": 0.95,
                            "status": "verified",
                            "evidence": match.group(0)[:200],
                            "reasoning": "Extracted via regex from policy text."
                        }
        return None
    
    @staticmethod
    def _extract_copay(text: str) -> Optional[Dict]:
        """Extract copay percentage - handles conditions."""
        patterns = [
            r'co[\s\-]?pay(?:ment)?\s*[:\-]\s*(\d+)\s*%([^\n\.]+)?',
            r'co[\s\-]?payment\s+of\s+(\d+)\s*%([^\n\.]+)?',
            r'(\d+)\s*%\s*co[\s\-]?pay([^\n\.]+)?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = int(match.group(1))
                # Check for conditions after the match
                condition = match.group(2) if len(match.groups()) > 1 and match.group(2) else ""
                
                # Look for condition words in the surrounding context
                context = text[max(0, match.start()-100):match.end()+200]
                condition_words = ['except', 'excluding', 'subject to', 'for senior', 'over age']
                has_condition = any(word in context.lower() for word in condition_words)
                
                return {
                    "value": value,
                    "display": f"{value}%" + (f" (with conditions)" if has_condition else ""),
                    "page": None,
                    "confidence": 0.95 if not has_condition else 0.80,
                    "status": "verified" if not has_condition else "requires_verification",
                    "evidence": match.group(0)[:200],
                    "reasoning": f"Extracted {value}% co-pay via regex. {'Conditions may apply.' if has_condition else ''}"
                }
        return None
    
    @staticmethod
    def _extract_grace_period(text: str) -> Optional[Dict]:
        """Extract grace period in days."""
        patterns = [
            r'[Gg]race\s+[Pp]eriod\s+of\s+(\d+)\s+days([^\n\.]+)?',
            r'[Gg]race\s+period\s*[:\-]\s*(\d+)\s+days([^\n\.]+)?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                value = int(match.group(1))
                return {
                    "value": value,
                    "display": f"{value} days",
                    "page": None,
                    "confidence": 0.95,
                    "status": "verified",
                    "evidence": match.group(0)[:200],
                    "reasoning": f"Extracted {value} days grace period via regex."
                }
        return None
    
    @staticmethod
    def _extract_free_look(text: str) -> Optional[Dict]:
        """Extract free look period in days."""
        patterns = [
            r'[Ff]ree\s+[Ll]ook\s+[Pp]eriod\s+of\s+(\d+)\s+days([^\n\.]+)?',
            r'[Ff]ree\s+[Ll]ook\s+period\s*[:\-]\s*(\d+)\s+days([^\n\.]+)?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                value = int(match.group(1))
                return {
                    "value": value,
                    "display": f"{value} days",
                    "page": None,
                    "confidence": 0.95,
                    "status": "verified",
                    "evidence": match.group(0)[:200],
                    "reasoning": f"Extracted {value} days free look period via regex."
                }
        return None
    
    @staticmethod
    def _extract_claim_settlement(text: str) -> Optional[Dict]:
        """Extract claim settlement days."""
        patterns = [
            r'[Cc]laim\s+[Ss]ettlement\s+within\s+(\d+)\s+days([^\n\.]+)?',
            r'[Cc]laims?\s+(?:are\s+)?settled\s+within\s+(\d+)\s+days([^\n\.]+)?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = int(match.group(1))
                return {
                    "value": value,
                    "display": f"{value} days",
                    "page": None,
                    "confidence": 0.95,
                    "status": "verified",
                    "evidence": match.group(0)[:200],
                    "reasoning": f"Extracted {value} days claim settlement via regex."
                }
        return None
    
    @staticmethod
    def _extract_conditional_fields(text: str) -> Dict[str, Dict]:
        """
        Extract fields that often have conditions.
        These get lower confidence and are flagged for verification.
        """
        results = {}
        
        # Room Rent - often has conditions
        room_match = re.search(
            r'[Rr]oom\s+[Rr]ent\s*(?:limit|sublimit)?\s*[:\-]\s*([^\n\.]+)',
            text
        )
        if room_match:
            value = room_match.group(1).strip()
            # Check for conditions
            has_condition = any(word in value.lower() for word in ['subject', 'except', 'excluding', 'senior'])
            results["room_rent_sublimit"] = {
                "value": value,
                "display": value,
                "page": None,
                "confidence": 0.80 if not has_condition else 0.65,
                "status": "verified" if not has_condition else "requires_verification",
                "evidence": room_match.group(0)[:200],
                "reasoning": f"Extracted via regex. {'Conditions detected: ' + value[:100] if has_condition else ''}"
            }
        
        # ICU - often has conditions
        icu_match = re.search(
            r'[Ii]CU\s*(?:limit|sublimit)?\s*[:\-]\s*([^\n\.]+)',
            text
        )
        if icu_match:
            value = icu_match.group(1).strip()
            has_condition = any(word in value.lower() for word in ['subject', 'except', 'excluding'])
            results["icu_sublimit"] = {
                "value": value,
                "display": value,
                "page": None,
                "confidence": 0.80 if not has_condition else 0.65,
                "status": "verified" if not has_condition else "requires_verification",
                "evidence": icu_match.group(0)[:200],
                "reasoning": f"Extracted via regex. {'Conditions may apply.' if has_condition else ''}"
            }
        
        return results