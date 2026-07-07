"""
summary_engine.py - Summary statistics computation
"""

from __future__ import annotations

from typing import Any, Dict, List


def compute_summary(results: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Compute summary statistics for all extracted attributes.
    """
    successful = []
    verified = []
    needs_verification = []
    not_found = []
    high_confidence = []
    medium_confidence = []
    low_confidence = []
    
    for attr_name, result in results.items():
        if attr_name.startswith("_"):
            continue
        
        value = result.get("value")
        status = result.get("status", "unknown")
        confidence = result.get("confidence", 0.0)
        
        if value is not None:
            successful.append(attr_name)
        
        if status == "verified":
            verified.append(attr_name)
        elif status == "requires_verification":
            needs_verification.append(attr_name)
        elif status == "not_found":
            not_found.append(attr_name)
        
        if confidence >= 0.8:
            high_confidence.append(attr_name)
        elif confidence >= 0.5:
            medium_confidence.append(attr_name)
        else:
            low_confidence.append(attr_name)
    
    total = len([k for k in results.keys() if not k.startswith("_")])
    
    return {
        "total_attributes": total,
        "successful_extractions": len(successful),
        "verified": len(verified),
        "requires_verification": len(needs_verification),
        "not_found": len(not_found),
        "success_rate": len(successful) / max(1, total),
        "confidence_breakdown": {
            "high": len(high_confidence),
            "medium": len(medium_confidence),
            "low": len(low_confidence)
        },
        "attributes_found": successful[:10],
        "attributes_verification_needed": needs_verification,
        "attributes_not_found": not_found
    }


def _extract_dynamic_compatibility(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Maintains backward compatibility by creating a _dynamic field
    from attributes that weren't found in the original implementation.
    """
    # Original dynamic attributes from the old code
    original_dynamic = [
        "policy_name", "insurer", "sum_insured_options", "policy_tenure",
        "lifetime_renewability", "copay_applicable", "copay_percentage",
        "room_rent_sublimit", "icu_sublimit", "inpatient_covered",
        "daycare_covered", "domiciliary_covered", "ambulance_covered",
        "organ_donor_covered", "cashless_available", "network_hospitals",
        "portability_available", "ncb_benefit"
    ]
    
    dynamic = {}
    for attr in original_dynamic:
        if attr in results:
            val = results[attr]
            if isinstance(val, dict):
                dynamic[attr] = val.get("value")
            else:
                dynamic[attr] = val
    
    return dynamic


__all__ = ["compute_summary", "_extract_dynamic_compatibility"]