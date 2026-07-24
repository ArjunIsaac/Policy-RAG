from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ConceptDefinition:
    name: str
    keywords: List[str]


CONCEPTS = [

    ConceptDefinition(
        name="pre_existing_disease",
        keywords=[
            "pre existing disease",
            "pre-existing disease",
            "ped",
        ],
    ),

    ConceptDefinition(
        name="specific_disease",
        keywords=[
            "specified disease",
            "specific disease",
            "listed condition",
            "listed illness",
        ],
    ),

    ConceptDefinition(
        name="waiting_period",
        keywords=[
            "waiting period",
        ],
    ),

    ConceptDefinition(
        name="premium",
        keywords=[
            "premium",
            "instalment",
            "installment",
        ],
    ),

    ConceptDefinition(
        name="grace_period",
        keywords=[
            "grace period",
        ],
    ),

    ConceptDefinition(
        name="renewal",
        keywords=[
            "renewal",
            "renew",
        ],
    ),

    ConceptDefinition(
        name="claim",
        keywords=[
            "claim",
            "cashless",
            "reimbursement",
        ],
    ),

]

def infer_concepts(heading, heading_full, heading_path, tags, text):
    searchable = " ".join([heading, heading_full, " ".join(heading_path)]).lower()
    matched = [c.name for c in CONCEPTS if any(k in searchable for k in c.keywords)]
    return sorted(set(matched))

def infer_query_concepts(query: str) -> list[str]:
    ql = query.lower()
    matched = []
    for concept in CONCEPTS:
        if any(keyword in ql for keyword in concept.keywords):
            matched.append(concept.name)
    return sorted(set(matched))