"""
post_processor.py - Validation, normalization, and cleaning
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from .config import CRITICAL_ATTRIBUTES


class PostProcessor:
    """Post-process and validate extracted attributes."""
    
    @staticmethod
    def validate_and_clean(results: Dict[str, Dict]) -> Dict[str, Dict]:
        """Validate extracted values and clean up."""
        cleaned = {}
        
        for attr_name, result in results.items():
            if attr_name not in CRITICAL_ATTRIBUTES:
                cleaned[attr_name] = result
                continue
            
            config = CRITICAL_ATTRIBUTES[attr_name]
            value = result.get("value")
            
            # Apply validation rules
            if config.validation_rules and value is not None:
                if config.type == "integer":
                    if "min" in config.validation_rules and value < config.validation_rules["min"]:
                        result["value"] = None
                        result["status"] = "requires_verification"
                        result["reasoning"] += f" | ⚠ Value below minimum ({config.validation_rules['min']})"
                    
                    if "max" in config.validation_rules and value > config.validation_rules["max"]:
                        result["value"] = None
                        result["status"] = "requires_verification"
                        result["reasoning"] += f" | ⚠ Value exceeds maximum ({config.validation_rules['max']})"
            
            # Ensure all fields exist
            for field in ["value", "display", "page", "clause", "evidence", "confidence", "status", "reasoning", "conflicts"]:
                if field not in result:
                    result[field] = None if field != "status" else "unknown"
                    if field == "conflicts":
                        result[field] = []
            
            cleaned[attr_name] = result
        
        return cleaned
    
    @staticmethod
    def normalize_value(value: Any, attr_type: str) -> Any:
        """Normalize value based on attribute type."""
        if value is None:
            return None
        
        try:
            if attr_type == "integer":
                if isinstance(value, (int, float)):
                    return int(value)
                elif isinstance(value, str):
                    nums = re.findall(r'\d+', value)
                    return int(nums[0]) if nums else None
            
            elif attr_type == "percentage":
                if isinstance(value, (int, float)):
                    return int(value)
                elif isinstance(value, str):
                    nums = re.findall(r'\d+', value)
                    return int(nums[0]) if nums else None
            
            elif attr_type == "boolean":
                if isinstance(value, bool):
                    return value
                elif isinstance(value, str):
                    return value.lower() in ("yes", "true", "y", "1", "available")
                return bool(value)
            
            elif attr_type == "list":
                if isinstance(value, list):
                    return value
                elif isinstance(value, str):
                    items = re.split(r'[,;|\n•]', value)
                    return [i.strip() for i in items if i.strip()]
                return [str(value)]
            
            elif attr_type == "duration":
                if isinstance(value, (int, float)):
                    return int(value)
                elif isinstance(value, str):
                    nums = re.findall(r'\d+', value)
                    return int(nums[0]) if nums else None
            
            else:  # string
                return str(value).strip()
        
        except Exception:
            return str(value)
    
    @staticmethod
    def get_display_value(value: Any, attr_type: str) -> str:
        """Get human-readable display value."""
        if value is None:
            return "Not specified in policy"
        
        if isinstance(value, bool):
            return "Yes" if value else "No"
        
        if isinstance(value, list):
            return "; ".join(str(v) for v in value) if value else "Not specified"
        
        if attr_type == "percentage" and isinstance(value, (int, float)):
            return f"{value}%"
        
        if attr_type == "integer" and isinstance(value, int):
            # Add context for waiting periods
            if "waiting" in attr_type or "days" in attr_type:
                if "days" in attr_type:
                    return f"{value} days"
                return f"{value} months"
            return str(value)
        
        return str(value)


__all__ = ["PostProcessor"]