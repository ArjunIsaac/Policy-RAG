"""
constants.py
------------
All configuration constants, prompts, and extraction schemas.
Edit this file to tune retrieval parameters, LLM prompts, or attribute groups.
"""

from __future__ import annotations

import nltk
from nltk.corpus import stopwords
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# ---------------------------------------------------------------------------
# Retrieval config
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-4B-AWQ"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
HYBRID_FETCH_K      = 30
FINAL_K             = 4
REORDER_ENABLED     = True

# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------

STOPWORDS = set(stopwords.words('english'))
CUSTOM_STOPWORDS = {
    'please', 'tell', 'me', 'know', 'want', 'ask', 'like', 'help',
    'thank', 'thanks', 'hi', 'hello', 'hey', 'maybe', 'perhaps',
    'basically', 'actually', 'really', 'quite', 'just', 'also', 'well',
    'look', 'see', 'think', 'guess', 'feel',
}
STOPWORDS.update(CUSTOM_STOPWORDS)

# ---------------------------------------------------------------------------
# Chat prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert insurance policy analyst reviewing regulatory policy contracts.

Your task is to analyze the retrieved context and provide a highly accurate determination of coverage, waiting periods, and exclusions.

CRITICAL LOGIC RULE (TERMINOLOGY MAPPING):
Users often ask questions using common language (e.g., "skin cancer", "eye surgery", "LASIK"). Insurance policies use formal legal or medical definitions (e.g., "skin carcinoma", "malignant melanoma", "cataract", "refractive error").
Before concluding that a condition is unmentioned, you MUST check if the common term maps to a formal definition or sub-exclusion within the context.

RULES:
1. If the answer is explicitly stated, answer directly.
2. Do NOT explain your reasoning.
3. Do NOT output any thinking, analysis, or intermediate steps.
4. Do NOT infer information that is not present.
5. If the policy does not mention the requested information, say:
   "The policy does not mention this."
6. If multiple conditions apply, summarize them clearly.
7. Quote the relevant wording whenever appropriate.
8. End every answer with the supporting citation(s), for example:
   (Page 13, Section: Pre-Existing Diseases)



RETRIEVED CONTEXT:
{context}
"""

CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])

# ---------------------------------------------------------------------------
# Attribute extraction schemas
# Each entry: (field_definitions_dict, rag_query, hint_for_llm)
# ---------------------------------------------------------------------------

PARTNER_ATTR_GROUPS = [
    (
        {
            "policy_name":          "string - product/plan name e.g. Total Health Plan",
            "insurer":              "string - insurance company name",
            "sum_insured_options":  "array of strings - all sum insured amounts e.g. ['5 Lakhs','10 Lakhs']",
            "policy_tenure":        "string - policy duration e.g. '1 Year'",
            "lifetime_renewability":"boolean - true if policy is lifetime renewable",
            "free_look_period_days":"number - free look period in days e.g. 15",
            "grace_period_days":    "number - grace period for renewal in days e.g. 30",
        },
        "policy name insurer sum insured tenure renewal free look grace period",
        "Look in: product name heading, sum insured table, tenure, renewal clause, free look period, grace period.",
    ),
    (
        {
            "waiting_period_initial_days":             "number - initial waiting period in days e.g. 30",
            "waiting_period_ped_months":               "number - pre-existing disease PED waiting period in months e.g. 48",
            "waiting_period_specific_illness_months":  "number - specific illness/procedure waiting period in months e.g. 24",
        },
        "initial waiting period pre-existing disease PED specific illness procedure waiting period",
        "Look in: Section C Waiting Period & Exclusions. Initial, PED, specific disease waiting periods.",
    ),
    (
        {
            "copay_applicable":   "boolean - true ONLY if a co-pay % is explicitly stated",
            "copay_percentage":   "number or null - co-pay % if stated, else null",
            "copay_conditions":   "string or null - when co-pay applies, else null",
            "room_rent_sublimit": "string or null - daily room rent cap if stated, else null",
            "icu_sublimit":       "string or null - ICU daily cap if stated, else null",
        },
        "co-payment copay room rent sub-limit ICU intensive care unit bed charges",
        "co-pay: only true if % explicitly stated. Room rent/ICU: only if a daily cap amount is mentioned.",
    ),
    (
        {
            "inpatient_covered":          "boolean - true if inpatient hospitalisation is covered",
            "daycare_covered":            "boolean - true if day care procedures are covered",
            "domiciliary_covered":        "boolean - true if domiciliary home treatment is covered",
            "maternity_covered":          "boolean - true if maternity expenses are covered",
            "ambulance_covered":          "boolean - true if emergency ambulance is covered",
            "organ_donor_covered":        "boolean - true if organ donor harvesting expenses are covered",
            "pre_hospitalisation_days":   "number - pre-hospitalisation cover in days e.g. 30",
            "post_hospitalisation_days":  "number - post-hospitalisation cover in days e.g. 60",
        },
        "inpatient day care domiciliary maternity ambulance organ donor pre-hospitalisation post-hospitalisation",
        "Look in: Section B Benefits table.",
    ),
    (
        {
            "cashless_available":    "boolean - true if cashless facility at network hospitals",
            "network_hospitals":     "string or null - count or description of network hospitals",
            "claim_settlement_days": "number - days insurer must settle claim e.g. 30",
            "portability_available": "boolean - true if policy can be ported to another insurer",
            "ncb_benefit":           "string or null - No Claim Bonus or Cumulative Bonus description",
        },
        "cashless network hospital claim settlement portability No Claim Bonus cumulative bonus NCB",
        "Look in: cashless service clause, claim settlement timeframe, portability, Cumulative Bonus.",
    ),
    (
        {
            "permanent_exclusions": "array of strings - key permanently excluded conditions (max 10)",
        },
        "permanent exclusions not covered excluded war cosmetic obesity adventure sports alcohol infertility",
        "Look in: Section C Standard and Specific General Exclusions.",
    ),
]

# Dynamic attributes prompt — one LLM call for policy-specific selling points
DYNAMIC_ATTR_PROMPT = """You are an expert insurance analyst helping insurance PARTNERS pitch policies to clients.

Read this insurance policy and identify ONLY benefits/features that:
1. Are a SELLING POINT a partner would highlight when pitching to a client
2. Are PRODUCT FEATURES — not definitions, not exclusions, not admin clauses

ONLY include things like:
- Restore/Recharge benefit (sum insured restored after a claim)
- Multiplier / Cumulative bonus (sum insured increases each claim-free year)
- OPD cover (outpatient consultations covered)
- Daily hospital cash benefit
- Newborn baby cover
- Mental health cover
- E-opinion / second medical opinion benefit
- Moratorium period (after X years, no pre-existing disease lookback)
- International cover
- Deductible options
- Health check-up benefit
- Any rider or add-on benefit

DO NOT include:
- Medical definitions (e.g. what TIA means, what dialysis means)
- Exclusions or what is NOT covered
- Admin clauses (fraud, nomination, cancellation, notices)
- Anything already in standard attributes (waiting periods, co-pay, room rent, maternity, exclusions)

CRITICAL RULES:
- NEVER invent a benefit.
- ONLY return a feature if explicit evidence exists in the supplied text.
- Return {{}} if no feature is explicitly found.

Return a JSON object: keys = snake_case feature names, values = short description with exact wording from the policy.

Policy text:
{text}

JSON:"""