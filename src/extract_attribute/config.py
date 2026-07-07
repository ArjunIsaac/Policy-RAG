"""
config.py - Attribute schemas and configuration
12 critical attributes for health insurance extraction
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AttributeConfig:
    """Configuration for a single attribute."""
    name: str
    question: str
    type: str  # "string", "integer", "boolean", "duration", "percentage", "list", "currency"
    alternatives: List[str] = field(default_factory=list)
    description: str = ""
    validation_rules: Dict[str, Any] = field(default_factory=dict)


# The 12 Most Critical Attributes for Health Insurance
CRITICAL_ATTRIBUTES: Dict[str, AttributeConfig] = {
    # === FINANCIAL & COVERAGE ===
    "sum_insured_options": AttributeConfig(
        name="sum_insured_options",
        question="What are the sum insured options available? List all amounts mentioned. (e.g., 5 Lakh, 10 Lakh, 25 Lakh)",
        type="list",
        alternatives=["coverage amounts", "policy limits", "sum assured", "coverage options"],
        description="The available coverage amounts the policy can be purchased for."
    ),
    
    "copay_percentage": AttributeConfig(
        name="copay_percentage",
        question="What is the co-payment percentage? (e.g., 10%, 20%)",
        type="percentage",
        alternatives=["co-pay percentage", "coinsurance percentage", "cost share percentage"],
        description="The percentage the policyholder must pay for each claim.",
        validation_rules={"min": 0, "max": 100}
    ),
    
    "waiting_period_ped_months": AttributeConfig(
        name="waiting_period_ped_months",
        question="What is the waiting period for Pre-Existing Diseases (PED) in months?",
        type="integer",
        alternatives=["pre-existing disease waiting", "PED waiting", "pre-existing condition waiting"],
        description="Waiting period before pre-existing diseases are covered.",
        validation_rules={"min": 0, "max": 60}
    ),
    
    "waiting_period_specific_illness_months": AttributeConfig(
        name="waiting_period_specific_illness_months",
        question="What is the waiting period for specific illnesses in months? (e.g., hernia, cataract, etc.)",
        type="integer",
        alternatives=["specific disease waiting", "listed conditions waiting", "specified illness waiting"],
        description="Waiting period for specific diseases mentioned in the policy.",
        validation_rules={"min": 0, "max": 60}
    ),
    
    "room_rent_sublimit": AttributeConfig(
        name="room_rent_sublimit",
        question="What is the room rent sub-limit? (e.g., 1% of SI, ₹5000/day, Single Private Room)",
        type="string",
        alternatives=["room limit", "accommodation limit", "room rent cap"],
        description="The room rent cap or restriction."
    ),
    
    "icu_sublimit": AttributeConfig(
        name="icu_sublimit",
        question="What is the ICU sub-limit? (e.g., 2% of SI, ₹10000/day)",
        type="string",
        alternatives=["intensive care limit", "ICU cap"],
        description="The ICU room charge cap."
    ),
    
    # === CLAIMS & BENEFITS ===
    "cashless_available": AttributeConfig(
        name="cashless_available",
        question="Is cashless claim facility available at network hospitals?",
        type="boolean",
        alternatives=["cashless treatment", "direct billing", "cashless facility"],
        description="Whether cashless treatment is offered."
    ),
    
    "claim_settlement_days": AttributeConfig(
        name="claim_settlement_days",
        question="What is the claim settlement timeline in days?",
        type="integer",
        alternatives=["claim processing time", "settlement timeline", "claim turnaround"],
        description="Number of days to settle claims.",
        validation_rules={"min": 0, "max": 90}
    ),
    
    "inpatient_covered": AttributeConfig(
        name="inpatient_covered",
        question="Are inpatient hospitalization expenses covered?",
        type="boolean",
        alternatives=["hospitalization coverage", "inpatient treatment"],
        description="Coverage for hospital stays exceeding 24 hours."
    ),
    
    "maternity_covered": AttributeConfig(
        name="maternity_covered",
        question="Are maternity expenses covered?",
        type="boolean",
        alternatives=["pregnancy coverage", "maternity benefits"],
        description="Whether maternity and childbirth expenses are covered."
    ),
    
    # === POLICY FEATURES ===
    "lifetime_renewability": AttributeConfig(
        name="lifetime_renewability",
        question="Is the policy renewable for life?",
        type="boolean",
        alternatives=["life long renewal", "permanent renewal", "lifetime renewal"],
        description="Whether the policy offers lifetime renewal."
    ),
    
    "portability_available": AttributeConfig(
        name="portability_available",
        question="Can this policy be ported to another insurer at renewal?",
        type="boolean",
        alternatives=["policy transfer", "portability option", "switch insurer"],
        description="Whether policy portability is offered."
    ),
}

# Keywords for scoring chunk relevance
RELEVANCE_KEYWORDS: Dict[str, List[str]] = {
    "sum_insured_options": ["sum insured", "coverage", "amount", "lakh", "crore", "option", "available"],
    "copay_percentage": ["copay", "co-pay", "coinsurance", "percentage", "share", "contribution"],
    "waiting_period_ped_months": ["pre-existing", "ped", "waiting", "months", "existing disease"],
    "waiting_period_specific_illness_months": ["specific", "listed", "disease", "waiting", "months", "condition"],
    "room_rent_sublimit": ["room rent", "accommodation", "room", "rent", "limit", "cap"],
    "icu_sublimit": ["icu", "intensive care", "critical care", "limit", "cap"],
    "cashless_available": ["cashless", "direct billing", "network", "tpa"],
    "claim_settlement_days": ["claim", "settlement", "processing", "days", "timeline"],
    "inpatient_covered": ["inpatient", "hospitalization", "admission", "covered"],
    "maternity_covered": ["maternity", "pregnancy", "childbirth", "delivery"],
    "lifetime_renewability": ["renew", "lifetime", "life long", "permanent"],
    "portability_available": ["port", "portability", "transfer", "switch"],
}

__all__ = ["AttributeConfig", "CRITICAL_ATTRIBUTES", "RELEVANCE_KEYWORDS"]