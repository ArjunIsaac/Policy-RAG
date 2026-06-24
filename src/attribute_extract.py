"""
attribute_extract.py
--------------------
Partner-focused policy attribute extraction.

Fixed attributes  → pure regex, instant, zero LLM calls.
Dynamic attributes → ONE LLM call on a small focused context.

This module is completely independent of the RAG chatbox pipeline.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from constants import DYNAMIC_ATTR_PROMPT

if TYPE_CHECKING:
    from vector_store import PolicyVectorStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean(s: str | None) -> str | None:
    """Strip markdown artifacts (**, *) and leading/trailing dashes."""
    if not s:
        return s
    s = re.sub(r"\*+", "", s)
    s = re.sub(r"^[\s\-–—]+", "", s)
    s = re.sub(r"[\s\-–—]+$", "", s)
    return s.strip() or None


def _find(patterns: list[str], text: str, flags: int = re.IGNORECASE) -> str | None:
    """Try each regex, return first captured group or None."""
    for pat in patterns:
        m = re.search(pat, text, flags)
        if m:
            try:
                return m.group(1).strip()
            except IndexError:
                return m.group(0).strip()
    return None


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    brace = raw.find("{")
    if brace > 0:
        raw = raw[brace:]
    end_brace = raw.rfind("}")
    if end_brace != -1 and end_brace < len(raw) - 1:
        raw = raw[:end_brace + 1]
    return raw.strip()


# ---------------------------------------------------------------------------
# Full text assembly from ChromaDB
# ---------------------------------------------------------------------------

def get_all_policy_text(store: "PolicyVectorStore", source_filter: list[str] | None) -> str:
    """Reassemble all chunks from ChromaDB sorted by page into one string."""
    try:
        col = store._client.get_collection(store.collection_name)
        where = None
        if source_filter and len(source_filter) == 1:
            where = {"source": {"$eq": source_filter[0]}}
        elif source_filter and len(source_filter) > 1:
            where = {"source": {"$in": source_filter}}

        result = (
            col.get(where=where, include=["documents", "metadatas"])
            if where
            else col.get(include=["documents", "metadatas"])
        )
        pairs = sorted(
            zip(result["documents"], result["metadatas"]),
            key=lambda x: x[1].get("page", 0),
        )
        seen, parts = set(), []
        for text, _ in pairs:
            if text not in seen:
                seen.add(text)
                parts.append(text)
        return "\n".join(parts)
    except Exception as e:
        print(f"[attr_extract] get_all_policy_text error: {e}")
        return ""


# ---------------------------------------------------------------------------
# Fixed attribute extraction — regex only
# ---------------------------------------------------------------------------

def extract_fixed_attributes(text: str) -> dict:
    """
    Extract all fixed partner-relevant attributes using regex patterns.
    Works across any Indian health insurance policy (IRDAI standard phrasing).
    Zero LLM calls — instant.
    """
    attrs: dict = {}

    # ── Policy Overview ───────────────────────────────────────────────────
    name = _find([r"Product\s+Name\s*[–—:-]+\s*([^\n\*]+)"], text)
    if not name:
        name = _find([
            r"(Total Health Plan|Health Guard|Optima Restore|Arogya Sanjeevani"
            r"|Star Comprehensive|Care Supreme|Niva Bupa[^\n]+)",
        ], text)
    attrs["policy_name"] = _clean(name)

    insurer = _find([
        r"^([A-Z][A-Za-z\s]+(?:General\s+)?Insurance\s+Company\s+Limited)",
    ], text, re.MULTILINE)
    if not insurer:
        insurer = _find([
            r"([A-Z][A-Za-z\s]{5,50}(?:General\s+)?Insurance\s+(?:Company\s+)?Limited)",
        ], text)
    attrs["insurer"] = _clean(insurer)

    # Sum insured — near "Sum Insured" label first, then fallback to all lakh mentions
    si_matches = re.findall(
        r"(?:Sum\s+Insured|sum\s+insured)[^\n]*?(\d+(?:\.\d+)?\s*(?:Lakh|Lakhs|Crore|Crores|L\b))",
        text, re.IGNORECASE,
    )
    if not si_matches:
        si_matches = re.findall(r"(\d+(?:\.\d+)?\s*Lakhs?\b)", text, re.IGNORECASE)
    attrs["sum_insured_options"] = list(dict.fromkeys(_clean(s) for s in si_matches)) or None

    tenure = _find([
        r"Tenure\s*[–—:-]+\s*([^\n\*]+)",
        r"[Pp]olicy\s+[Pp]eriod[^\n]*?(\d+\s*(?:Year|Years|Month|Months))",
        r"tenure\s+of\s+(\d+\s*(?:Year|Years|Month|Months))",
    ], text)
    attrs["policy_tenure"] = _clean(tenure)

    attrs["lifetime_renewability"] = bool(re.search(
        r"ordinarily\s+be\s+renewable|lifetime\s+renew|renewable\s+for\s+life",
        text, re.IGNORECASE,
    ))

    fl = _find([
        r"free\s+look\s+period\s+of\s+(fifteen|\d+)\s+days",
        r"free\s+look[^\n]*?(fifteen|\d+)\s+days",
    ], text)
    _word_num = {"fifteen": 15, "thirty": 30, "seven": 7, "fourteen": 14}
    attrs["free_look_period_days"] = (
        _word_num.get(fl.lower(), int(fl) if fl and fl.isdigit() else None)
        if fl else None
    )

    gp = _find([
        r"[Gg]race\s+[Pp]eriod\s+of\s+(thirty|\d+)\s+days",
        r"[Gg]race\s+[Pp]eriod[^\n]*?(\d+)\s+days",
        r"renewed\s+within\s+(?:the\s+)?[Gg]race\s+[Pp]eriod\s+of\s+(\d+)\s+days",
    ], text)
    attrs["grace_period_days"] = (
        _word_num.get(gp.lower(), int(gp) if gp and gp.isdigit() else None)
        if gp else None
    )

    # ── Waiting Periods ───────────────────────────────────────────────────
    wp_init = _find([
        r"(\d+)[\s-]*[Dd]ay\s+[Ww]aiting\s+[Pp]eriod",
        r"within\s+(\d+)\s+days\s+from\s+the\s+first\s+policy\s+commencement",
        r"treatment\s+of\s+any\s+illness\s+within\s+(\d+)\s+days",
    ], text)
    attrs["waiting_period_initial_days"] = int(wp_init) if wp_init and wp_init.isdigit() else None

    # PED: find the number inside "expiry of X months" near pre-existing context
    ped_m = re.search(
        r"pre.existing[^\n]{0,200}expiry\s+of\s+(\d+)\s+months",
        text, re.IGNORECASE | re.DOTALL,
    )
    if not ped_m:
        ped_m = re.search(
            r"expiry\s+of\s+(\d+)\s+months\s+of\s+continuous\s+coverage\s+after[^\n]{0,100}"
            r"(?:pre.existing|PED|first\s+policy)",
            text, re.IGNORECASE,
        )
    if not ped_m:
        ped_m = re.search(
            r"(?:pre.existing|PED)[^\n]{0,100}(\d+)\s+months\s+of\s+continuous",
            text, re.IGNORECASE,
        )
    attrs["waiting_period_ped_months"] = int(ped_m.group(1)) if ped_m else None

    # Specific illness: separate clause from PED, near "listed Conditions" or "Excl02"
    spec_m = re.search(
        r"(?:listed\s+[Cc]onditions?|specific\s+(?:disease|illness|procedure)|Excl02)"
        r"[^\n]{0,300}expiry\s+of\s+(\d+)\s+months",
        text, re.IGNORECASE | re.DOTALL,
    )
    if not spec_m:
        spec_m = re.search(
            r"expiry\s+of\s+(\d+)\s+months\s+of\s+continuous\s+coverage\s+after[^\n]{0,100}"
            r"(?:listed|inception\s+of\s+the\s+first)",
            text, re.IGNORECASE,
        )
    if spec_m:
        spec_val = int(spec_m.group(1))
        ped_val = attrs.get("waiting_period_ped_months")
        # Avoid echoing PED value; look for a distinct second number if same
        if ped_val and spec_val == ped_val:
            alt = re.search(
                r"expiry\s+of\s+(\d+)\s+months\s+of\s+continuous\s+coverage",
                text[text.find(spec_m.group(0)) + len(spec_m.group(0)):],
                re.IGNORECASE,
            )
            attrs["waiting_period_specific_illness_months"] = int(alt.group(1)) if alt else None
        else:
            attrs["waiting_period_specific_illness_months"] = spec_val
    else:
        attrs["waiting_period_specific_illness_months"] = None

    # ── Co-pay & Sub-limits ───────────────────────────────────────────────
    copay_m = re.search(r"co[\s-]?pay(?:ment)?[^\n]*?(\d+)\s*%", text, re.IGNORECASE)
    if copay_m:
        attrs["copay_applicable"] = True
        attrs["copay_percentage"] = int(copay_m.group(1))
        attrs["copay_conditions"] = _clean(_find([r"co[\s-]?pay[^\n]{10,200}"], text))
    else:
        attrs["copay_applicable"] = False
        attrs["copay_percentage"] = None
        attrs["copay_conditions"] = None

    attrs["room_rent_sublimit"] = _find([
        r"[Rr]oom\s+[Rr]ent\s+[Ss]ub.?[Ll]imit\s*[:\-–]?\s*([^\n]+)",
        r"[Rr]oom\s+[Rr]ent[^\n]*?(\d+%\s+of\s+[Ss]um\s+[Ii]nsured[^\n]*)",
    ], text)

    attrs["icu_sublimit"] = _find([
        r"ICU\s+[Ss]ub.?[Ll]imit\s*[:\-–]?\s*([^\n]+)",
        r"[Ii]ntensive\s+[Cc]are[^\n]*?(\d+%\s+of\s+[Ss]um\s+[Ii]nsured[^\n]*)",
    ], text)

    # ── Coverage ──────────────────────────────────────────────────────────
    attrs["inpatient_covered"]   = bool(re.search(r"[Ii]n.?patient\s+[Tt]reatment", text))
    attrs["daycare_covered"]     = bool(re.search(r"[Dd]ay\s+[Cc]are\s+(?:Procedures?|[Tt]reatments?)", text))
    attrs["domiciliary_covered"] = bool(re.search(r"[Dd]omiciliary\s+[Tt]reatment", text))
    attrs["maternity_covered"]   = bool(re.search(
        r"[Mm]aternity\s+(?:[Ee]xpense|[Tt]reatment|[Bb]enefit|[Cc]over)", text))
    attrs["ambulance_covered"]   = bool(re.search(r"[Ee]mergency\s+[Aa]mbulance", text))
    attrs["organ_donor_covered"] = bool(re.search(r"[Oo]rgan\s+[Dd]onor", text))

    pre = _find([r"[Pp]re.hospitalisation[^\n]*?(\d+)\s+days"], text)
    attrs["pre_hospitalisation_days"] = int(pre) if pre and pre.isdigit() else None

    post = _find([r"[Pp]ost.hospitalisation[^\n]*?(\d+)\s+days"], text)
    attrs["post_hospitalisation_days"] = int(post) if post and post.isdigit() else None

    # ── Claims & Renewals ─────────────────────────────────────────────────
    attrs["cashless_available"] = bool(re.search(
        r"[Cc]ashless\s+(?:[Ff]acility|[Ss]ervice)", text))

    # Network hospitals — only capture if a substantial number is mentioned
    nh = _find([r"(\d[\d,]+\+?\s*[Nn]etwork\s+[Hh]ospitals?)"], text)
    if not nh:
        nh = _find([r"[Nn]etwork\s+of\s+(\d[\d,]+\+?\s*[Hh]ospitals?)"], text)
    attrs["network_hospitals"] = _clean(nh)

    cs = _find([r"settle\s+or\s+reject\s+a\s+claim[^\n]*?within\s+(\d+)\s+days"], text)
    attrs["claim_settlement_days"] = int(cs) if cs and cs.isdigit() else None

    attrs["portability_available"] = bool(re.search(
        r"[Pp]ortability|port\s+the\s+policy", text))

    ncb = _find([
        r"[Cc]umulative\s+[Bb]onus\s+means\s+([^\n]{20,200})",
        r"[Nn]o\s+[Cc]laim\s+[Bb]onus\s+means\s+([^\n]{20,200})",
    ], text)
    attrs["ncb_benefit"] = _clean(ncb)

    # ── Permanent Exclusions ──────────────────────────────────────────────
    excl_m = re.search(
        r"(?:Standard\s+General\s+Exclusions?|2\.\s+Standard\s+General)(.*?)"
        r"(?:3\.\s+Specific|Section\s+D|\Z)",
        text, re.IGNORECASE | re.DOTALL,
    )
    if excl_m:
        excl_text = excl_m.group(1)
        excl_items = re.findall(
            r"(?:^|\n)\s*(?:i{1,3}v?|vi*|ix|x{1,3}|[a-z])\)\s*([A-Z][^\n:]{10,100})",
            excl_text,
        )
        if not excl_items:
            excl_items = re.findall(r"\*\*([^*\n]{10,100})\*\*", excl_text)
        attrs["permanent_exclusions"] = list(dict.fromkeys(excl_items[:12])) or None
    else:
        attrs["permanent_exclusions"] = None

    return attrs


# ---------------------------------------------------------------------------
# Dynamic attributes — one LLM call
# ---------------------------------------------------------------------------

def extract_dynamic_attributes(llm, parser, text: str) -> dict:
    """
    ONE LLM call on focused policy text to find partner-relevant selling points.
    Uses first 6000 chars — benefit sections are always early in the document.
    """
    focused = text[:6000]
    prompt = DYNAMIC_ATTR_PROMPT.format(text=focused)
    try:
        raw = parser.invoke(llm.invoke([
            SystemMessage(content="You are an insurance analyst. Respond with ONLY valid JSON. No explanation."),
            HumanMessage(content=prompt),
        ]))
        raw = _strip_fences(raw)
        result = json.loads(raw)
        if isinstance(result, dict):
            return result
    except Exception as e:
        print(f"[attr_extract] Dynamic attributes error: {e}")
    return {}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_extraction(store: "PolicyVectorStore", llm, parser, source_filter: list[str] | None) -> dict:
    """
    Full attribute extraction pipeline.
    Fixed fields: regex (instant).
    Dynamic fields: 1 LLM call (~30s).
    """
    text = get_all_policy_text(store, source_filter)
    if not text:
        return {"_error": "Could not retrieve policy text from store"}

    attrs = extract_fixed_attributes(text)
    found = sum(1 for v in attrs.values() if v is not None)
    print(f"[attr_extract] Regex extraction: {found}/{len(attrs)} fields found")

    dynamic = extract_dynamic_attributes(llm, parser, text)
    if dynamic:
        attrs["_dynamic"] = dynamic

    return attrs