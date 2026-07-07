"""
extraction_engine.py - Batch extraction with single LLM call
Extracts ONLY the fields that regex didn't handle
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from langchain_core.messages import HumanMessage, SystemMessage

from .config import CRITICAL_ATTRIBUTES
from .retrieval_engine import RetrievedChunk
from .post_processor import PostProcessor

if TYPE_CHECKING:
    pass


class BatchExtractor:
    """
    Extracts attributes in a single LLM call.
    Supports field filtering - only extract what's needed.
    """
    
    def __init__(self, llm, parser):
        self.llm = llm
        self.parser = parser
        self._field_filter = None  # Set of field names to extract
        self.system_prompt = """You are a precise insurance policy analyst.
Extract ONLY the requested attributes from the provided context.
Return ONLY valid JSON. No explanations, no markdown.
Never guess or invent values - use null if not found."""
    
    def extract_all(
        self, 
        allocated_chunks: Dict[str, List[RetrievedChunk]], 
        field_names: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """
        Extract attributes with ONE LLM call.
        If field_names is provided, only extract those.
        """
        # Step 1: Filter fields if specified
        if field_names:
            # Only keep chunks for fields we need
            filtered_chunks = {}
            for attr_name in field_names:
                if attr_name in allocated_chunks:
                    filtered_chunks[attr_name] = allocated_chunks[attr_name]
            self._field_filter = set(field_names)
        else:
            filtered_chunks = allocated_chunks
            self._field_filter = set(CRITICAL_ATTRIBUTES.keys())
        
        # If no fields to extract, return empty
        if not filtered_chunks or not any(filtered_chunks.values()):
            print("[BatchExtractor] No fields to extract with LLM")
            return {attr_name: self._empty_result("No chunks found") for attr_name in self._field_filter}
        
        # Step 2: Build prompt with only requested fields
        prompt = self._build_prompt(filtered_chunks)
        
        print(f"[BatchExtractor] Extracting {len(self._field_filter)} fields with LLM")
        print(f"[BatchExtractor] Prompt length: {len(prompt)} chars, ~{len(prompt)//4} tokens")
        
        try:
            print("[BatchExtractor] Making single LLM call...")
            
            response = self.llm.invoke([
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt)
            ])
            
            raw = self.parser.invoke(response)
            
            # Clean and parse
            cleaned = self._clean_response(raw)
            extracted_data = self._parse_response(cleaned)
            
            if not extracted_data:
                print("[BatchExtractor] Failed to parse JSON, attempting repair...")
                extracted_data = self._repair_json(raw)
            
            # Normalize values
            normalized = {}
            for attr_name in self._field_filter:
                if attr_name in extracted_data and extracted_data[attr_name]:
                    normalized[attr_name] = self._normalize_value(
                        extracted_data[attr_name],
                        CRITICAL_ATTRIBUTES[attr_name],
                        allocated_chunks.get(attr_name, [])
                    )
                else:
                    normalized[attr_name] = self._empty_result(
                        f"Attribute {attr_name} not found in LLM response"
                    )
            
            return normalized
            
        except Exception as e:
            print(f"[BatchExtractor] Error: {e}")
            import traceback
            traceback.print_exc()
            
            # Return empty results for all requested fields
            return {
                attr_name: self._empty_result(f"Extraction error: {str(e)}")
                for attr_name in self._field_filter
            }
    
    def _build_context(self, allocated_chunks: Dict[str, List[RetrievedChunk]]) -> str:
        """Build context with SMART truncation to fit 4K limit."""
        context_parts = []
        chunk_index = 1
        max_chunks = 12  # 1 per attribute max
        max_chars = 600  # ~150 tokens per chunk
        
        for attr_name, chunks in allocated_chunks.items():
            if not chunks or chunk_index > max_chunks:
                continue
            
            # Only include if this field is in our filter
            if self._field_filter and attr_name not in self._field_filter:
                continue
            
            # Use best chunk only
            best_chunk = chunks[0]
            content = best_chunk.content
            
            # Smart truncation: keep beginning and any numbers
            if len(content) > max_chars:
                beginning = content[:400]
                numbers = re.findall(r'\d+[\s,]*\d*[\s]*(?:Lakh|Crore|%|days|months|years|Rs\.?|₹)', content[400:])
                if numbers:
                    number_context = " [Key values: " + ", ".join(numbers[:5]) + "]"
                    content = beginning + number_context
                else:
                    truncated = content[:max_chars]
                    last_space = truncated.rfind(' ')
                    content = truncated[:last_space] + "..." if last_space > 0 else truncated + "..."
            
            context_parts.append(f"\n--- {attr_name} ---")
            context_parts.append(f"[Page {best_chunk.page}] {content}")
            context_parts.append("")
            chunk_index += 1
        
        return "\n".join(context_parts)
    
    def _build_prompt(self, allocated_chunks: Dict[str, List[RetrievedChunk]]) -> str:
        """Build extraction prompt with ONLY the requested fields."""
        context = self._build_context(allocated_chunks)
        
        # Determine which fields to extract
        if self._field_filter:
            attr_names = self._field_filter
        else:
            attr_names = set(CRITICAL_ATTRIBUTES.keys())
        
        # Build per-attribute guidance
        attr_guidance_lines = []
        for attr_name in sorted(attr_names):
            if attr_name in CRITICAL_ATTRIBUTES:
                config = CRITICAL_ATTRIBUTES[attr_name]
                desc = config.description if config.description else f"Extract the {attr_name}"
                attr_guidance_lines.append(f"- {attr_name}: {desc}")
        attr_guidance = "\n".join(attr_guidance_lines)
        
        # Build JSON fields
        fields = []
        for attr_name in sorted(attr_names):
            fields.append(f'    "{attr_name}": null')
        fields_str = ",\n".join(fields)
        
        return f"""Context:
{context}

Extract these attributes from the context above. Return ONLY valid JSON.

Attributes to extract:
{attr_guidance}

For each attribute, provide:
- value: the extracted value (use null if not found)
- page: the page number where you found it
- clause: the section/clause name
- evidence: a short verbatim snippet

Format:
{{
    "sum_insured_options": {{"value": ["5 Lakh", "10 Lakh"], "page": 4, "clause": "Schedule", "evidence": "Sum Insured options are 5 Lakh, 10 Lakh"}},
    "copay_percentage": {{"value": 20, "page": 15, "clause": "Co-pay Clause", "evidence": "A 20% co-payment applies"}},
    ... (all requested attributes)
}}

CRITICAL RULES:
1. If an attribute is not found, set value to null
2. For numeric values, extract ONLY the number (e.g., 24 not "24 months")
3. For percentages, extract ONLY the number (e.g., 20 not "20%")
4. For lists, use array format ["item1", "item2"]
5. For booleans, use true or false (lowercase)
6. NEVER guess or invent values
7. Return ONLY valid JSON - no explanations, no markdown
8. Use commas between all fields and objects
9. DO NOT add a comma after the last field

JSON:"""
    
    def _clean_response(self, raw: str) -> str:
        """Clean LLM response before parsing."""
        raw = raw.strip()
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return raw
    
    def _parse_response(self, raw: str) -> Dict[str, Any]:
        """Parse JSON response from LLM."""
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"[Parser] JSON parse error: {e}")
            return {}
    
    def _repair_json(self, raw: str) -> Dict[str, Any]:
        """Robust JSON repair - handles missing commas, trailing commas, unescaped quotes."""
        result = {}
        
        # Try to find each attribute's complete object
        for attr_name in self._field_filter:
            # Look for pattern with flexible matching
            pattern = rf'"{attr_name}"\s*:\s*\{{\s*"value"\s*:\s*([^,}}]+)'
            match = re.search(pattern, raw, re.DOTALL)
            
            if match:
                attr_obj = {"value": None, "page": None, "clause": None, "evidence": None}
                
                # Extract and parse value
                val_str = match.group(1).strip()
                
                if val_str == 'null':
                    attr_obj["value"] = None
                elif val_str.startswith('['):
                    try:
                        list_str = val_str.replace("'", '"')
                        list_str = re.sub(r',\s*,', ', null,', list_str)
                        attr_obj["value"] = json.loads(list_str)
                    except:
                        items = re.findall(r'"([^"]+)"', val_str)
                        attr_obj["value"] = items if items else None
                elif val_str.startswith('"'):
                    val_cleaned = val_str.strip('"')
                    val_cleaned = val_cleaned.replace('\\"', '"')
                    attr_obj["value"] = val_cleaned
                elif val_str.lower() in ('true', 'false'):
                    attr_obj["value"] = val_str.lower() == 'true'
                else:
                    try:
                        attr_obj["value"] = int(val_str) if '.' not in val_str else float(val_str)
                    except:
                        attr_obj["value"] = val_str
                
                # Extract other fields
                for field in ["page", "clause", "evidence"]:
                    field_match = re.search(rf'"{field}"\s*:\s*"([^"]*)"', raw[match.start():match.end()+300])
                    if field_match:
                        if field == "page":
                            try:
                                attr_obj[field] = int(field_match.group(1))
                            except:
                                pass
                        else:
                            attr_obj[field] = field_match.group(1)
                
                result[attr_name] = attr_obj
            else:
                result[attr_name] = {"value": None}
        
        return result
    
    def _normalize_value(self, extracted: Dict[str, Any], config, chunks: List[RetrievedChunk]) -> Dict[str, Any]:
        """Normalize extracted value based on attribute type."""
        if not extracted:
            return self._empty_result("No extraction data")
        
        # Handle both formats
        if "value" in extracted:
            raw_value = extracted["value"]
            page = extracted.get("page")
            clause = extracted.get("clause")
            evidence = extracted.get("evidence")
        else:
            raw_value = extracted
            page = None
            clause = None
            evidence = None
        
        if raw_value is None:
            return self._empty_result("Not found in context")
        
        normalized = raw_value
        
        try:
            if config.type == "integer":
                if isinstance(raw_value, (int, float)):
                    normalized = int(raw_value)
                elif isinstance(raw_value, str):
                    nums = re.findall(r'\d+', raw_value)
                    normalized = int(nums[0]) if nums else None
            
            elif config.type == "percentage":
                if isinstance(raw_value, (int, float)):
                    normalized = int(raw_value)
                elif isinstance(raw_value, str):
                    nums = re.findall(r'\d+', raw_value)
                    normalized = int(nums[0]) if nums else None
            
            elif config.type == "boolean":
                if isinstance(raw_value, bool):
                    normalized = raw_value
                elif isinstance(raw_value, str):
                    normalized = raw_value.lower() in ("yes", "true", "y", "1", "available", "included", "covered")
                else:
                    normalized = bool(raw_value)
            
            elif config.type == "list":
                if isinstance(raw_value, list):
                    normalized = raw_value
                elif isinstance(raw_value, str):
                    try:
                        normalized = json.loads(raw_value)
                    except:
                        items = re.split(r'[,;|\n•]', raw_value)
                        normalized = [i.strip() for i in items if i.strip()]
                else:
                    normalized = [str(raw_value)]
            
            else:  # string
                normalized = str(raw_value).strip()
        
        except Exception as e:
            print(f"[Normalize] Error for {config.name}: {e}")
            normalized = None
        
        # Compute confidence
        confidence = 0.85 if normalized is not None else 0.0
        if chunks and normalized is not None:
            confidence = min(0.95, 0.85 + (len(chunks) * 0.03))
        
        display = PostProcessor.get_display_value(normalized, config.type)
        
        return {
            "value": normalized,
            "display": display,
            "page": page,
            "clause": clause,
            "evidence": evidence,
            "confidence": confidence,
            "status": "verified" if normalized is not None else "not_found",
            "reasoning": self._build_reasoning(normalized, config, chunks),
            "conflicts": []
        }
    
    def _build_reasoning(self, value: Any, config, chunks: List[RetrievedChunk]) -> str:
        if value is None:
            return f"Could not find {config.name} in the policy context."
        
        display = PostProcessor.get_display_value(value, config.type)
        parts = [f"Extracted value: {display}"]
        
        if chunks:
            pages = sorted(set(c.page for c in chunks if c.page))
            if pages:
                parts.append(f"Found on page(s): {', '.join(str(p) for p in pages)}")
        
        return " | ".join(parts)
    
    def _empty_result(self, reason: str = "") -> Dict[str, Any]:
        return {
            "value": None,
            "display": "Not specified in policy",
            "page": None,
            "clause": None,
            "evidence": None,
            "confidence": 0.0,
            "status": "not_found",
            "reasoning": f"Not found: {reason}" if reason else "Not found in policy",
            "conflicts": []
        }


__all__ = ["BatchExtractor"]