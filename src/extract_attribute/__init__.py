"""
extract_attribute - Public interface
Maintains backward compatibility with existing code.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from datetime import datetime

from .config import CRITICAL_ATTRIBUTES
from .retrieval_engine import SmartRetriever
from .extraction_engine import BatchExtractor
from .post_processor import PostProcessor
from .summary_engine import compute_summary, _extract_dynamic_compatibility
from .regex_extractor import RegexExtractor

if TYPE_CHECKING:
    from vector_store import PolicyVectorStore


def run_extraction(
    store: "PolicyVectorStore",
    llm,
    parser,
    source_filter: List[str] = None,
) -> Dict[str, Any]:
    """
    MAIN ENTRY POINT - Maintains backward compatibility.
    Uses hybrid approach: Regex + LLM
    """
    print("[AttrExtract] Starting hybrid extraction...")
    print(f"[AttrExtract] Target: {len(CRITICAL_ATTRIBUTES)} critical attributes")
    
    try:
        # Step 1: Get full text for regex extraction
        print("[AttrExtract] Step 1: Regex extraction (0 LLM calls)...")
        text, page_map = store.get_all_policy_text(source_filter) if hasattr(store, 'get_all_policy_text') else ("", [])
        
        # Step 2: Regex extraction
        regex_results = RegexExtractor.extract_all(text) if text else {}
        
        # Convert regex results to citation envelope format
        for attr_name, result in regex_results.items():
            if "page" not in result or result["page"] is None:
                # Try to find page from page_map
                if page_map:
                    # Simple heuristic: first page
                    result["page"] = page_map[0][1] if page_map else 1
            # Add empty conflicts
            if "conflicts" not in result:
                result["conflicts"] = []
        
        print(f"[AttrExtract] Regex extracted: {len(regex_results)} fields")
        
        # Step 3: LLM extraction for remaining fields
        print("[AttrExtract] Step 2: LLM extraction for complex fields...")
        
        # Get only the fields that regex didn't handle
        regex_field_names = set(regex_results.keys())
        all_field_names = set(CRITICAL_ATTRIBUTES.keys())
        llm_field_names = all_field_names - regex_field_names
        
        print(f"[AttrExtract] LLM will handle: {len(llm_field_names)} fields")
        
        # Run LLM extraction with reduced field set
        llm_results = run_llm_extraction(store, llm, parser, source_filter, llm_field_names)
        
        # Step 4: Combine results (regex takes precedence for overlapping fields)
        combined = {**llm_results, **regex_results}  # regex overrides LLM
        
        # Step 5: Compute summary
        print("[AttrExtract] Step 3: Computing summary...")
        summary = compute_summary(combined)
        combined["_summary"] = summary
        
        # Add dynamic compatibility field
        combined["_dynamic"] = _extract_dynamic_compatibility(combined)
        
        # Add metadata
        combined["_metadata"] = {
            "extraction_timestamp": datetime.utcnow().isoformat(),
            "llm_calls": 1,
            "regex_fields": list(regex_field_names),
            "llm_fields": list(llm_field_names),
            "attributes_extracted": len(CRITICAL_ATTRIBUTES),
            "source_filter": source_filter,
            "hardware": "RTX 3050 (6GB VRAM)",
            "model": "Qwen3-4B"
        }
        
        print(f"[AttrExtract] ✅ Complete! Found {summary['successful_extractions']}/{summary['total_attributes']} attributes")
        print(f"[AttrExtract] ✅ Regex: {len(regex_field_names)} fields, LLM: {len(llm_field_names)} fields")
        
        return combined
        
    except Exception as e:
        print(f"[AttrExtract] ❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "_error": str(e),
            "_summary": {
                "total_attributes": len(CRITICAL_ATTRIBUTES),
                "successful_extractions": 0,
                "error": str(e)
            }
        }


def run_llm_extraction(store, llm, parser, source_filter, field_names=None) -> Dict[str, Dict]:
    """
    Run LLM extraction for specified fields.
    If field_names is None, extract all fields.
    """
    # Get allocated chunks
    retriever = SmartRetriever(store, top_k=20)
    allocated_chunks = retriever.retrieve_and_route(source_filter)
    
    # Extract with LLM
    extractor = BatchExtractor(llm, parser)
    results = extractor.extract_all(allocated_chunks, field_names)
    
    return results


__all__ = ["run_extraction"]