"""
attribute_extract.py
--------------------
Partner-focused policy attribute extraction.

Fixed attributes  → pure regex, instant, zero LLM calls.
Dynamic attributes → ONE LLM call on a small focused context.

Every extracted field is wrapped in a citation envelope:
  {
      "value":      <extracted value or None>,
      "display":    <human-readable string for UI>,
      "page":       <page number or None>,
      "clause":     <section/clause label or None>,
      "confidence": <"high" | "medium" | "low" | "not_found">,
      "status":     <"verified" | "not_specified" | "requires_verification">,
      "reasoning":  <one-sentence explanation of how the value was derived>,
      "evidence":   <verbatim snippet from policy text, or None>,
      "conflicts":  <list of {value, page, clause, evidence} dicts if multiple values found, else []>
  }
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
    if not s:
        return s
    s = re.sub(r"\*+", "", s)
    s = re.sub(r"^[\s\-\u2013\u2014]+", "", s)
    s = re.sub(r"[\s\-\u2013\u2014]+$", "", s)
    return s.strip() or None


def _find_m(patterns: list, text: str, flags: int = re.IGNORECASE):
    """Returns the first match object across all patterns."""
    for pat in patterns:
        m = re.search(pat, text, flags | re.DOTALL)
        if m:
            return m
    return None


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    # Strip qwen3 <think>...</think> blocks
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    brace = raw.find("{")
    if brace > 0:
        raw = raw[brace:]
    end_brace = raw.rfind("}")
    if end_brace != -1 and end_brace < len(raw) - 1:
        raw = raw[: end_brace + 1]
    return raw.strip()


def _char_to_page(char_pos: int, page_map: list) -> int | None:
    if not page_map:
        return None
    page = page_map[0][1]
    for offset, pg in page_map:
        if char_pos >= offset:
            page = pg
        else:
            break
    return page


def _extract_snippet(text: str, match_start: int, match_end: int, window: int = 200) -> str:
    """
    Extract a human-readable evidence snippet around a regex match.
    Trims to sentence boundaries where possible, max `window` chars each side.
    """
    start = max(0, match_start - window)
    end   = min(len(text), match_end + window)
    snippet = text[start:end].strip()
    # Clean up whitespace / newlines for display
    snippet = re.sub(r"\s+", " ", snippet)
    # Add ellipsis if truncated
    if start > 0:
        snippet = "…" + snippet
    if end < len(text):
        snippet = snippet + "…"
    return snippet


# ---------------------------------------------------------------------------
# Citation envelope
# ---------------------------------------------------------------------------

def _field(
    value,
    display: str | None = None,
    page: int | None = None,
    clause: str | None = None,
    confidence: str = "high",
    status: str = "verified",
    reasoning: str | None = None,
    evidence: str | None = None,
    conflicts: list | None = None,
) -> dict:
    if display is None:
        if value is None:
            display = "Not specified in policy"
        elif isinstance(value, bool):
            display = "Yes" if value else "No"
        elif isinstance(value, list):
            display = "; ".join(str(v) for v in value) if value else "Not specified in policy"
        else:
            display = str(value)
    return {
        "value":     value,
        "display":   display,
        "page":      page,
        "clause":    clause,
        "confidence": confidence,
        "status":    status,
        "reasoning": reasoning,
        "evidence":  evidence,
        "conflicts": conflicts or [],
    }


def _not_found(reasoning: str | None = None) -> dict:
    return _field(None, "Not specified in policy",
                  confidence="not_found", status="not_specified",
                  reasoning=reasoning or "No matching clause or keyword found in policy text.")


def _not_applicable(reasoning: str | None = None) -> dict:
    return _field(None, "Not applicable",
                  confidence="high", status="verified",
                  reasoning=reasoning or "Not applicable given other policy conditions.")


# ---------------------------------------------------------------------------
# Full text assembly from ChromaDB
# ---------------------------------------------------------------------------

def get_all_policy_text(
    store: "PolicyVectorStore",
    source_filter: list | None,
) -> tuple:
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
        seen: set = set()
        parts: list = []
        page_map: list = []
        cursor = 0
        for chunk_text, meta in pairs:
            if chunk_text not in seen:
                seen.add(chunk_text)
                page_map.append((cursor, int(meta.get("page", 0))))
                parts.append(chunk_text)
                cursor += len(chunk_text) + 1

        return "\n".join(parts), page_map
    except Exception as e:
        print(f"[attr_extract] get_all_policy_text error: {e}")
        return "", []


# ---------------------------------------------------------------------------
# Fixed attribute extraction
# ---------------------------------------------------------------------------

_WORD_NUM = {
    "fifteen": 15, "thirty": 30, "seven": 7, "fourteen": 14,
    "ten": 10, "twenty": 20, "forty": 40, "sixty": 60, "ninety": 90,
}


def extract_fixed_attributes(text: str, page_map: list) -> dict:
    attrs: dict = {}
    pm = page_map

    def pg(m) -> int | None:
        return _char_to_page(m.start(), pm) if m else None

    def snip(m) -> str | None:
        return _extract_snippet(text, m.start(), m.end()) if m else None

    # ── Policy Overview ────────────────────────────────────────────────────

    # Policy name — try explicit label first (high confidence), then known name list (high too,
    # since a name match in the document is unambiguous), only fall to medium if truly uncertain.
    m = _find_m([r"Product\s+Name\s*[–\-—:-]+\s*([^\n\*]+)"], text)
    if m:
        attrs["policy_name"] = _field(
            _clean(m.group(1)), page=pg(m), clause="Product Name", confidence="high",
            reasoning="Extracted from explicit 'Product Name' label in policy schedule.",
            evidence=snip(m),
        )
    else:
        m2 = _find_m([
            r"(Total Health Plan|Health Guard|Optima Restore|Arogya Sanjeevani"
            r"|Star Comprehensive|Care Supreme|Niva Bupa[^\n]+)"
        ], text)
        # Known product name found in document → still high confidence; name is unambiguous.
        attrs["policy_name"] = (
            _field(_clean(m2.group(1)), page=pg(m2), confidence="high",
                   reasoning="Policy name matched from known product name list in document; no explicit 'Product Name' label present.",
                   evidence=snip(m2))
            if m2 else _not_found("No 'Product Name' label or known policy name pattern found.")
        )

    m = _find_m([r"^([A-Z][A-Za-z\s]+(?:General\s+)?Insurance\s+Company\s+Limited)"], text, re.MULTILINE)
    if not m:
        m = _find_m([r"([A-Z][A-Za-z\s]{5,50}(?:General\s+)?Insurance\s+(?:Company\s+)?Limited)"], text)
    attrs["insurer"] = (
        _field(_clean(m.group(1)), page=pg(m), clause="Preamble", confidence="high",
               reasoning="Insurer name matched via 'Insurance Company Limited' pattern.",
               evidence=snip(m))
        if m else _not_found("No 'Insurance Company Limited' pattern found.")
    )

    si_matches = re.findall(
        r"(?:Sum\s+Insured|sum\s+insured)[^\n]*?(\d+(?:\.\d+)?\s*(?:Lakh|Lakhs|Crore|Crores|L\b))",
        text, re.IGNORECASE,
    )
    if not si_matches:
        si_matches = re.findall(r"(\d+(?:\.\d+)?\s*Lakhs?\b)", text, re.IGNORECASE)
    si_list = list(dict.fromkeys(_clean(s) for s in si_matches)) or None
    m_si = _find_m([r"[Ss]um\s+[Ii]nsured[^\n]{0,60}?\d"], text)
    attrs["sum_insured_options"] = _field(
        si_list,
        display=", ".join(si_list) if si_list else "Not specified in policy",
        page=pg(m_si),
        clause="Schedule / Product Details",
        confidence="high" if si_list else "not_found",
        status="verified" if si_list else "not_specified",
        reasoning=(
            f"Found {len(si_list)} sum insured option(s) via 'Sum Insured' keyword scan."
            if si_list else "No sum insured figures found near 'Sum Insured' keyword."
        ),
        evidence=snip(m_si),
    )

    m = _find_m([
        r"Tenure\s*[–\-—:-]+\s*([^\n\*]+)",
        r"[Pp]olicy\s+[Pp]eriod[^\n]*?(\d+\s*(?:Year|Years|Month|Months))",
    ], text)
    attrs["policy_tenure"] = (
        _field(_clean(m.group(1)), page=pg(m), clause="Schedule", confidence="high",
               reasoning="Tenure extracted from policy schedule label.", evidence=snip(m))
        if m else _not_found("No 'Tenure' or 'Policy Period' label found.")
    )

    m_renew = re.search(
        r"ordinarily\s+be\s+renewable|lifetime\s+renew|renewable\s+for\s+life",
        text, re.IGNORECASE,
    )
    attrs["lifetime_renewability"] = _field(
        bool(m_renew),
        page=_char_to_page(m_renew.start(), pm) if m_renew else None,
        clause="Renewal Clause",
        confidence="high" if m_renew else "medium",
        status="verified",
        reasoning=(
            "Lifetime renewability phrase explicitly found in renewal clause."
            if m_renew else "No explicit lifetime renewability phrase found; defaulting to False."
        ),
        evidence=_extract_snippet(text, m_renew.start(), m_renew.end()) if m_renew else None,
    )

    m = _find_m([
        r"[Ff]ree\s+[Ll]ook\s+[Pp]eriod\s+of\s+(fifteen|\d+)\s+days",
        r"[Ff]ree\s+[Ll]ook[^\n]*?(fifteen|\d+)\s+days",
    ], text)
    if m:
        raw_fl = m.group(1)
        val_fl = _WORD_NUM.get(raw_fl.lower(), int(raw_fl) if raw_fl.isdigit() else None)
        attrs["free_look_period_days"] = _field(
            val_fl, display=f"{val_fl} days" if val_fl else None,
            page=pg(m), clause="Free Look Period", confidence="high",
            reasoning="Duration extracted from 'Free Look Period' clause.", evidence=snip(m),
        )
    else:
        attrs["free_look_period_days"] = _not_found("No 'Free Look Period' clause found.")

    m = _find_m([
        r"[Gg]race\s+[Pp]eriod\s+of\s+(thirty|\d+)\s+days",
        r"[Gg]race\s+[Pp]eriod[^\n]*?(\d+)\s+days",
        r"renewed\s+within\s+(?:the\s+)?[Gg]race\s+[Pp]eriod\s+of\s+(\d+)\s+days",
    ], text)
    if m:
        raw_gp = m.group(1)
        val_gp = _WORD_NUM.get(raw_gp.lower(), int(raw_gp) if raw_gp.isdigit() else None)
        attrs["grace_period_days"] = _field(
            val_gp, display=f"{val_gp} days" if val_gp else None,
            page=pg(m), clause="Grace Period", confidence="high",
            reasoning="Duration extracted from 'Grace Period' clause.", evidence=snip(m),
        )
    else:
        attrs["grace_period_days"] = _not_found("No 'Grace Period' clause found.")

    # ── Waiting Periods ────────────────────────────────────────────────────

    m = _find_m([
        r"(\d+)[\s-]*[Dd]ay\s+[Ww]aiting\s+[Pp]eriod",
        r"within\s+(\d+)\s+days\s+from\s+the\s+first\s+policy\s+commencement",
        r"treatment\s+of\s+any\s+illness\s+within\s+(\d+)\s+days",
    ], text)
    if m:
        v = int(m.group(1))
        attrs["waiting_period_initial_days"] = _field(
            v, display=f"{v} days",
            page=pg(m), clause="Section C – Standard Waiting Period (Excl03)",
            confidence="high",
            reasoning=f"Explicitly stated as {v}-day initial waiting period in Excl03.",
            evidence=snip(m),
        )
    else:
        attrs["waiting_period_initial_days"] = _not_found(
            "No initial waiting period clause (Excl03) found."
        )

    # ── PED Waiting Period — full conflict detection ────────────────────────
    # Collect every occurrence across all patterns, preserving clause label and snippet.
    # If multiple distinct values exist, surface ALL of them as conflicts rather than
    # silently choosing one.

    _ped_patterns = [
        (
            r"[Pp]re[\s\-]?[Ee]xisting\s+[Dd]isease[^\)]{0,500}?expiry\s+of\s+(\d+)\s+months",
            "Pre-Existing Disease keyword + 'expiry of N months'"
        ),
        (
            r"[Cc]ode[\s\-]*Excl01[^\)]{0,600}?expiry\s+of\s+(\d+)\s+months",
            "Clause Excl01"
        ),
        (
            r"expiry\s+of\s+(\d+)\s+months\s+of\s+continuous\s+coverage\s+after\s+the\s+date"
            r"\s+of\s+inception\s+of\s+the\s+first\s+[Pp]olicy\s+with\s+[Uu]s",
            "'expiry of N months of continuous coverage after inception'"
        ),
    ]

    ped_candidates: list[dict] = []
    for pat, label in _ped_patterns:
        for match in re.finditer(pat, text, re.DOTALL | re.IGNORECASE):
            try:
                months = int(match.group(1))
                ped_candidates.append({
                    "value":    months,
                    "page":     _char_to_page(match.start(), pm),
                    "clause":   label,
                    "evidence": _extract_snippet(text, match.start(), match.end()),
                })
                print(f"[attr_extract][PED] Candidate: {months} months via '{label}' p.{_char_to_page(match.start(), pm)}")
            except (IndexError, ValueError):
                pass

    if ped_candidates:
        unique_vals = list(dict.fromkeys(c["value"] for c in ped_candidates))

        if len(unique_vals) > 1:
            # ── CONFLICT: surface all candidates, do not silently resolve ──
            print(f"[attr_extract][PED] ⚠ Conflict detected: {unique_vals}")
            # Use the value from Excl01 as the primary if present, else the minimum
            excl01_candidates = [c for c in ped_candidates if "Excl01" in c["clause"]]
            primary = excl01_candidates[0] if excl01_candidates else min(ped_candidates, key=lambda x: x["value"])
            conflict_list = [c for c in ped_candidates if c != primary]

            attrs["waiting_period_ped_months"] = _field(
                primary["value"],
                display=f"{primary['value']} months",
                page=primary["page"],
                clause=primary["clause"],
                confidence="medium",
                status="requires_verification",
                reasoning=(
                    f"⚠ Multiple waiting periods detected: {unique_vals} months. "
                    f"Showing value from {primary['clause']}. "
                    f"Manual verification required — see conflicts."
                ),
                evidence=primary["evidence"],
                conflicts=conflict_list,
            )
        else:
            # Single value — high confidence
            best = ped_candidates[0]
            attrs["waiting_period_ped_months"] = _field(
                best["value"],
                display=f"{best['value']} months",
                page=best["page"],
                clause=best["clause"],
                confidence="high",
                status="verified",
                reasoning=f"Consistently stated as {best['value']} months across all matching clauses.",
                evidence=best["evidence"],
            )
    else:
        attrs["waiting_period_ped_months"] = _not_found(
            "No PED waiting period clause (Excl01 or equivalent) found."
        )

    # Specific illness waiting period
    spec_candidates: list = []
    for pat in [
        r"[Ll]isted\s+[Cc]onditions?[^\)]{0,600}?expiry\s+of\s+(\d+)\s+months",
        r"[Cc]ode[\s\-]*Excl02[^\)]{0,600}?expiry\s+of\s+(\d+)\s+months",
        r"[Ss]pecified\s+disease[^\)]{0,400}?expiry\s+of\s+(\d+)\s+months",
    ]:
        for match in re.finditer(pat, text, re.DOTALL | re.IGNORECASE):
            try:
                spec_candidates.append((int(match.group(1)), match.start(), match.end()))
            except (IndexError, ValueError):
                pass

    if spec_candidates:
        ped_v = attrs["waiting_period_ped_months"]["value"]
        candidates = [(v, s, e) for v, s, e in spec_candidates if ped_v is None or v <= ped_v]
        if candidates:
            best_v, best_s, best_e = min(candidates, key=lambda x: x[0])
            attrs["waiting_period_specific_illness_months"] = _field(
                best_v, display=f"{best_v} months",
                page=_char_to_page(best_s, pm),
                clause="Section C – Specified Disease Waiting Period (Excl02)",
                confidence="high",
                reasoning=f"Explicitly stated as {best_v} months in Clause Excl02.",
                evidence=_extract_snippet(text, best_s, best_e),
            )
        else:
            attrs["waiting_period_specific_illness_months"] = _not_found(
                "Specific illness values found but all exceed PED period — verify."
            )
    else:
        attrs["waiting_period_specific_illness_months"] = _not_found(
            "No specific illness waiting period clause (Excl02) found."
        )

    # ── Co-pay & Sub-limits ────────────────────────────────────────────────

    copay_m = re.search(r"co[\s\-]?pay(?:ment)?[^\n]*?(\d+)\s*%", text, re.IGNORECASE)
    if copay_m:
        pct = int(copay_m.group(1))
        attrs["copay_applicable"] = _field(
            True, display="Yes",
            page=_char_to_page(copay_m.start(), pm),
            clause="Co-payment Clause", confidence="high",
            reasoning=f"Co-payment of {pct}% explicitly stated.",
            evidence=_extract_snippet(text, copay_m.start(), copay_m.end()),
        )
        attrs["copay_percentage"] = _field(
            pct, display=f"{pct}%",
            page=_char_to_page(copay_m.start(), pm),
            confidence="high",
            reasoning=f"Percentage ({pct}%) extracted directly from co-payment clause.",
            evidence=_extract_snippet(text, copay_m.start(), copay_m.end()),
        )
        cond_m = _find_m([r"co[\s\-]?pay[^\n]{10,200}"], text)
        attrs["copay_conditions"] = _field(
            _clean(cond_m.group(0)) if cond_m else None,
            page=pg(cond_m) if cond_m else None,
            confidence="medium",
            reasoning="Co-payment conditions extracted from surrounding clause text.",
            evidence=snip(cond_m),
        )
    else:
        attrs["copay_applicable"]  = _field(False, display="No", confidence="high", status="verified",
                                             reasoning="No co-payment percentage pattern found; treated as no co-pay.")
        attrs["copay_percentage"]  = _not_applicable("Co-pay not applicable.")
        attrs["copay_conditions"]  = _not_applicable("Co-pay not applicable.")

    # Room rent sub-limit
    rrm = _find_m([
        r"[Rr]oom\s+[Rr]ent\s+[Ss]ub[\s\-]?[Ll]imit\s*[:\-–]?\s*([^\n]+)",
        r"[Rr]oom\s+[Rr]ent[^\n]*?(\d+%\s+of\s+[Ss]um\s+[Ii]nsured[^\n]*)",
        r"[Rr]oom\s+[Rr]ent[^\n]*?(Rs\.?\s*\d[\d,]*\s*(?:per\s+day)?[^\n]{0,40})",
    ], text)
    if rrm:
        attrs["room_rent_sublimit"] = _field(
            _clean(rrm.group(1)), page=pg(rrm),
            clause="Room Rent Sub-limit", confidence="high",
            reasoning="Explicit room rent cap found in policy wording.",
            evidence=snip(rrm),
        )
    else:
        if re.search(r"[Rr]oom\s+[Rr]ent", text):
            attrs["room_rent_sublimit"] = _field(
                None,
                display="No sub-limit specified (full coverage likely — verify policy schedule)",
                confidence="low", status="requires_verification",
                reasoning="'Room Rent' mentioned in document but no explicit cap (% of SI or Rs./day) found.",
            )
        else:
            attrs["room_rent_sublimit"] = _not_found("'Room Rent' not mentioned in document.")

    icum = _find_m([
        r"ICU\s+[Ss]ub[\s\-]?[Ll]imit\s*[:\-–]?\s*([^\n]+)",
        r"[Ii]ntensive\s+[Cc]are[^\n]*?(\d+%\s+of\s+[Ss]um\s+[Ii]nsured[^\n]*)",
        r"ICU[^\n]*?(Rs\.?\s*\d[\d,]*\s*(?:per\s+day)?[^\n]{0,40})",
    ], text)
    if icum:
        attrs["icu_sublimit"] = _field(
            _clean(icum.group(1)), page=pg(icum),
            clause="ICU Sub-limit", confidence="high",
            reasoning="Explicit ICU cap found in policy wording.",
            evidence=snip(icum),
        )
    else:
        if re.search(r"\bICU\b|[Ii]ntensive\s+[Cc]are\s+[Uu]nit", text):
            attrs["icu_sublimit"] = _field(
                None,
                display="No sub-limit specified (full coverage likely — verify policy schedule)",
                confidence="low", status="requires_verification",
                reasoning="ICU mentioned in document but no explicit cap found.",
            )
        else:
            attrs["icu_sublimit"] = _not_found("ICU/Intensive Care not mentioned in document.")

    # ── Coverage ───────────────────────────────────────────────────────────

    def _bool_coverage(patterns, clause_label, found_r, not_found_r):
        m = _find_m(patterns, text)
        return _field(
            bool(m), page=pg(m), clause=clause_label,
            confidence="high" if m else "medium",
            reasoning=found_r if m else not_found_r,
            evidence=snip(m),
        )

    attrs["inpatient_covered"] = _bool_coverage(
        [r"[Ii]n[\-\s]?patient\s+[Tt]reatment"],
        "Section B – Inpatient Benefits",
        "Inpatient treatment explicitly listed as a covered benefit.",
        "Inpatient treatment keyword not found; coverage not confirmed.",
    )
    attrs["daycare_covered"] = _bool_coverage(
        [r"[Dd]ay\s+[Cc]are\s+(?:[Pp]rocedures?|[Tt]reatments?)"],
        "Section B – Inpatient Benefits",
        "Day care procedures explicitly listed as covered.",
        "Day care keyword not found; coverage not confirmed.",
    )
    attrs["domiciliary_covered"] = _bool_coverage(
        [r"[Dd]omiciliary\s+[Tt]reatment"],
        "Section B – Inpatient Benefits",
        "Domiciliary treatment explicitly listed as covered.",
        "Domiciliary treatment keyword not found; coverage not confirmed.",
    )

    # Maternity — check benefits then cross-check exclusions
    mat_benefit_m = re.search(
        r"[Mm]aternity\s+(?:[Ee]xpense|[Tt]reatment|[Bb]enefit|[Cc]over)", text
    )
    mat_excl_m = re.search(
        r"(?:Excl18|[Mm]aternity\s*:\s*[Cc]ode\s*[-–]\s*Excl18|"
        r"[Ss]tandard\s+[Gg]eneral\s+[Ee]xclusion[^\n]{0,200}[Mm]aternity)",
        text, re.IGNORECASE | re.DOTALL
    )
    if mat_benefit_m and not mat_excl_m:
        attrs["maternity_covered"] = _field(
            True, page=_char_to_page(mat_benefit_m.start(), pm),
            clause="Section B – Benefits", confidence="high",
            reasoning="Maternity benefit found with no corresponding exclusion clause (Excl18).",
            evidence=_extract_snippet(text, mat_benefit_m.start(), mat_benefit_m.end()),
        )
    elif mat_benefit_m and mat_excl_m:
        attrs["maternity_covered"] = _field(
            False,
            display="Excluded in base policy (Code Excl18) — available as add-on only",
            page=_char_to_page(mat_excl_m.start(), pm),
            clause="Section C – Standard General Exclusions (Excl18)",
            confidence="high", status="verified",
            reasoning="Maternity appears in both benefits (definition) and exclusions (Excl18); exclusion takes precedence for base policy.",
            evidence=_extract_snippet(text, mat_excl_m.start(), mat_excl_m.end()),
        )
    else:
        attrs["maternity_covered"] = _not_found("Maternity keyword not found in benefits section.")

    m = _find_m([r"[Ee]mergency\s+[Aa]mbulance"], text)
    attrs["ambulance_covered"] = _field(
        bool(m), page=pg(m), clause="Section B – Emergency Ambulance",
        confidence="high" if m else "medium",
        reasoning="Emergency ambulance explicitly listed." if m else "Ambulance keyword not found.",
        evidence=snip(m),
    )

    m = _find_m([r"[Oo]rgan\s+[Dd]onor"], text)
    attrs["organ_donor_covered"] = _field(
        bool(m), page=pg(m), clause="Section B – Organ Donor",
        confidence="high" if m else "medium",
        reasoning="Organ donor cover explicitly listed." if m else "Organ donor keyword not found.",
        evidence=snip(m),
    )

    m = _find_m([r"[Pp]re[\-\s]hospitalisation[^\n]*?(\d+)\s+days"], text)
    if m:
        v = int(m.group(1))
        attrs["pre_hospitalisation_days"] = _field(
            v, display=f"{v} days",
            page=pg(m), clause="Section B – Pre-hospitalisation", confidence="high",
            reasoning=f"Pre-hospitalisation period of {v} days explicitly stated.",
            evidence=snip(m),
        )
    else:
        attrs["pre_hospitalisation_days"] = _not_found("No pre-hospitalisation period found.")

    m = _find_m([r"[Pp]ost[\-\s]hospitalisation[^\n]*?(\d+)\s+days"], text)
    if m:
        v = int(m.group(1))
        attrs["post_hospitalisation_days"] = _field(
            v, display=f"{v} days",
            page=pg(m), clause="Section B – Post-hospitalisation", confidence="high",
            reasoning=f"Post-hospitalisation period of {v} days explicitly stated.",
            evidence=snip(m),
        )
    else:
        attrs["post_hospitalisation_days"] = _not_found("No post-hospitalisation period found.")

    # ── Claims & Renewals ──────────────────────────────────────────────────

    m = _find_m([r"[Cc]ashless\s+(?:[Ff]acility|[Ss]ervice)"], text)
    attrs["cashless_available"] = _field(
        bool(m), page=pg(m), clause="Cashless Service",
        confidence="high" if m else "medium",
        reasoning="Cashless facility explicitly mentioned." if m else "Cashless facility keyword not found.",
        evidence=snip(m),
    )

    nh_m = _find_m([
        r"(\d[\d,]{2,}\+?\s*[Nn]etwork\s+[Hh]ospitals?)",
        r"[Nn]etwork\s+of\s+(\d[\d,]{2,}\+?\s*[Hh]ospitals?)",
        r"[Nn]etwork\s+[Hh]ospitals?[^\n]{0,40}?(\d[\d,]{2,}\+?)",
    ], text)
    if nh_m:
        attrs["network_hospitals"] = _field(
            _clean(nh_m.group(1)), page=pg(nh_m),
            clause="Network Provider", confidence="high",
            reasoning="Network hospital count explicitly stated in policy.",
            evidence=snip(nh_m),
        )
    else:
        attrs["network_hospitals"] = _field(
            None,
            display="Not stated in policy document — refer insurer's website for current network",
            confidence="not_found", status="requires_verification",
            reasoning="No numeric hospital count found; insurer website should be checked.",
        )

    m = _find_m([r"settle\s+or\s+reject\s+a\s+claim[^\n]*?within\s+(\d+)\s+days"], text)
    if m:
        v = int(m.group(1))
        attrs["claim_settlement_days"] = _field(
            v, display=f"{v} days from receipt of last required document",
            page=pg(m), clause="Section D – Provision for Penal Interest",
            confidence="high",
            reasoning=f"Claim settlement timeline of {v} days explicitly stated.",
            evidence=snip(m),
        )
    else:
        attrs["claim_settlement_days"] = _not_found("No claim settlement timeline clause found.")

    m = _find_m([r"[Pp]ortability|port\s+the\s+policy"], text)
    attrs["portability_available"] = _field(
        bool(m), page=pg(m), clause="Section D – Portability",
        confidence="high" if m else "medium",
        reasoning="Portability explicitly mentioned." if m else "Portability keyword not found.",
        evidence=snip(m),
    )

    # NCB
    ncb_pct_m = re.search(r"[Cc]umulative\s+[Bb]onus[^\n]{0,150}?(\d+\s*%[^\n]{0,80})", text)
    if not ncb_pct_m:
        ncb_pct_m = re.search(
            r"(\d+\s*%)\s*(?:of\s+[Ss]um\s+[Ii]nsured\s+)?(?:per|for\s+each)\s+claim[\s\-]free",
            text, re.IGNORECASE
        )
    if ncb_pct_m:
        attrs["ncb_benefit"] = _field(
            _clean(ncb_pct_m.group(1)),
            page=_char_to_page(ncb_pct_m.start(), pm),
            clause="Cumulative Bonus", confidence="high",
            reasoning="NCB percentage explicitly stated in Cumulative Bonus clause.",
            evidence=_extract_snippet(text, ncb_pct_m.start(), ncb_pct_m.end()),
        )
    else:
        ncb_def_m = re.search(r"[Cc]umulative\s+[Bb]onus\s+means", text)
        if ncb_def_m:
            attrs["ncb_benefit"] = _field(
                "Increase in Sum Insured without increase in premium",
                display="Applicable — exact percentage not stated (check policy schedule)",
                page=_char_to_page(ncb_def_m.start(), pm),
                clause="Def. 10 – Cumulative Bonus",
                confidence="low", status="requires_verification",
                reasoning="Cumulative Bonus defined in policy but no percentage stated; schedule must be checked.",
                evidence=_extract_snippet(text, ncb_def_m.start(), ncb_def_m.end()),
            )
        else:
            attrs["ncb_benefit"] = _not_found("No Cumulative Bonus clause or definition found.")

    # ── Permanent Exclusions ───────────────────────────────────────────────

    EXCL_CODES = {
        "Excl01": "Pre-Existing Diseases",
        "Excl02": "Specified Disease / Procedure Waiting Period",
        "Excl03": "Initial 30-Day Waiting Period",
        "Excl04": "Investigation & Evaluation Only",
        "Excl05": "Rest Cure / Rehabilitation / Respite Care",
        "Excl06": "Obesity / Weight Control",
        "Excl07": "Change-of-Gender Treatments",
        "Excl08": "Cosmetic or Plastic Surgery",
        "Excl09": "Hazardous / Adventure Sports",
        "Excl10": "Breach of Law",
        "Excl11": "Excluded Providers",
        "Excl12": "Alcoholism / Drug / Substance Abuse",
        "Excl13": "Health Hydros / Nature Cure / Spas",
        "Excl14": "Dietary Supplements without Prescription",
        "Excl15": "Refractive Error Correction < 7.5 Dioptres",
        "Excl16": "Unproven / Experimental Treatments",
        "Excl17": "Sterility and Infertility",
        "Excl18": "Maternity Expenses",
    }

    found_excl_codes = []
    first_excl_page = None
    first_excl_evidence = None
    for code, label in EXCL_CODES.items():
        cm = re.search(re.escape(code), text, re.IGNORECASE)
        if cm:
            found_excl_codes.append(label)
            if first_excl_page is None:
                first_excl_page = _char_to_page(cm.start(), pm)
                first_excl_evidence = _extract_snippet(text, cm.start(), cm.end())

    excl_section_m = re.search(
        r"(?:3\.\s*Specific\s+General\s+Exclusions?|[Ss]pecific\s+[Gg]eneral\s+[Ee]xclusions?)"
        r"(.*?)(?:[Ss]ection\s+D|\Z)",
        text, re.IGNORECASE | re.DOTALL,
    )
    specific_excl: list = []
    if excl_section_m:
        excl_text = excl_section_m.group(1)
        items = re.findall(
            r"(?:^|\n)\s*(?:[ivxlcdm]+\)|[a-z]\)|[ivx]+\.)\s*([A-Z][^\n]{15,120})",
            excl_text, re.MULTILINE
        )
        if not items:
            items = re.findall(
                r"(?:^|\n)\s*([A-Z][A-Za-z\s,/]{15,100})(?:\.|$)",
                excl_text, re.MULTILINE
            )
        specific_excl = list(dict.fromkeys(i.strip() for i in items[:8]))

    all_excl = list(dict.fromkeys(found_excl_codes + specific_excl))

    if all_excl:
        attrs["permanent_exclusions"] = _field(
            all_excl,
            display="\n• " + "\n• ".join(all_excl),
            page=first_excl_page,
            clause="Section C – Standard & Specific General Exclusions",
            confidence="high" if found_excl_codes else "medium",
            status="verified",
            reasoning=f"{len(all_excl)} exclusion(s) identified via IRDAI standard Excl codes in Section C.",
            evidence=first_excl_evidence,
        )
    else:
        attrs["permanent_exclusions"] = _field(
            None,
            display="Not specified in policy (refer Section C)",
            confidence="not_found", status="requires_verification",
            reasoning="No IRDAI Excl codes or exclusion section text found.",
        )

    return attrs


# ---------------------------------------------------------------------------
# Policy Risk Summary — computed from extracted attrs
# ---------------------------------------------------------------------------

def compute_policy_summary(attrs: dict) -> dict:
    """
    Derive a top-level risk summary from extracted attributes.
    Returned dict is consumed by the UI to render the summary card.
    """
    def val(key):
        v = attrs.get(key, {})
        return v.get("value") if isinstance(v, dict) else v

    def display(key):
        v = attrs.get(key, {})
        return v.get("display", "Not specified") if isinstance(v, dict) else str(v)

    def conf(key):
        v = attrs.get(key, {})
        return v.get("confidence", "not_found") if isinstance(v, dict) else "not_found"

    # Count high-confidence fields (excluding exclusions list and dynamic)
    core_keys = [k for k in attrs if not k.startswith("_")]
    total = len(core_keys)
    high = sum(1 for k in core_keys if conf(k) == "high")
    medium = sum(1 for k in core_keys if conf(k) == "medium")
    not_found = sum(1 for k in core_keys if conf(k) == "not_found")
    overall_confidence = round((high * 1.0 + medium * 0.5) / total * 100) if total else 0

    # Coverage tier
    has_copay       = val("copay_applicable") is True
    has_room_limit  = val("room_rent_sublimit") is not None
    has_maternity   = val("maternity_covered") is True
    ped_months      = val("waiting_period_ped_months")
    copay_pct       = val("copay_percentage")

    if not has_copay and not has_room_limit:
        coverage_tier = "Comprehensive"
    elif has_copay and copay_pct and copay_pct >= 20:
        coverage_tier = "Basic"
    else:
        coverage_tier = "Standard"

    # Conflicts present?
    conflicts_found = []
    for k, v in attrs.items():
        if isinstance(v, dict) and v.get("conflicts"):
            conflicts_found.append(k)

    return {
        "coverage_tier":        coverage_tier,
        "ped_waiting_period":   display("waiting_period_ped_months"),
        "copay":                display("copay_applicable") if not has_copay else f"{copay_pct}%",
        "room_rent":            display("room_rent_sublimit"),
        "portability":          "Available" if val("portability_available") else "Not specified",
        "renewability":         "Lifetime" if val("lifetime_renewability") else "Not specified",
        "maternity":            "Included" if has_maternity else display("maternity_covered"),
        "overall_confidence":   overall_confidence,
        "fields_high":          high,
        "fields_medium":        medium,
        "fields_not_found":     not_found,
        "fields_total":         total,
        "conflicts":            conflicts_found,
    }


# ---------------------------------------------------------------------------
# Dynamic attributes — one LLM call
# ---------------------------------------------------------------------------

def extract_dynamic_attributes(llm, parser, text: str) -> dict:
    focused = text[:6000]
    print(f"[attr_extract] Dynamic input length: {len(focused)} chars")
    prompt = DYNAMIC_ATTR_PROMPT.format(text=focused)
    try:
        raw = parser.invoke(llm.invoke([
            SystemMessage(content=(
                "You are an insurance analyst. "
                "Respond with ONLY valid JSON. No preamble, no explanation, no markdown fences. "
                "Do not include any reasoning or thinking text."
            )),
            HumanMessage(content=prompt),
        ]))
        raw = _strip_fences(raw)
        result = json.loads(raw)
        if isinstance(result, dict):
            skip_kw = ("not found", "null", "none", "n/a", "-", "discount",
                       "loading", "cancellation", "fraud", "nomination")
            return {
                k: v for k, v in result.items()
                if v and not any(kw in str(v).lower() for kw in skip_kw)
            }
    except Exception as e:
        print(f"[attr_extract] Dynamic attributes error: {e}")
    return {}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_extraction(
    store: "PolicyVectorStore",
    llm,
    parser,
    source_filter: list | None,
) -> dict:
    text, page_map = get_all_policy_text(store, source_filter)
    if not text:
        return {"_error": "Could not retrieve policy text from store"}

    attrs = extract_fixed_attributes(text, page_map)
    found = sum(
        1 for v in attrs.values()
        if isinstance(v, dict) and v.get("value") is not None
    )
    print(f"[attr_extract] Regex extraction: {found}/{len(attrs)} fields populated")

    dynamic = extract_dynamic_attributes(llm, parser, text)
    if dynamic:
        attrs["_dynamic"] = dynamic

    # Compute and attach top-level summary
    attrs["_summary"] = compute_policy_summary(attrs)

    return attrs