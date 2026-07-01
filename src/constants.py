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
