"""
Two baselines, required by the brief, so the LLM agent's numbers mean
something relative to a floor:

1. TRIVIAL baseline: majority-class intent, always auto-handle, one
   generic templated reply regardless of message content. This is the
   "what if we did nothing smart at all" floor.

2. SIMPLE baseline: keyword/regex-based intent classifier (no LLM) +
   nearest-neighbor-only reply (no generation, just return the most
   similar historical brand reply verbatim) + fixed confidence proxy
   for escalation. This is the "what if we did the obvious non-LLM thing"
   comparison point.
"""
import re
from collections import Counter


def trivial_baseline_fit(golden_intents: list) -> str:
    """Returns the majority intent label from the golden set's true labels."""
    return Counter(golden_intents).most_common(1)[0][0]


def trivial_baseline_predict(message: str, majority_intent: str, generic_reply: str) -> dict:
    return {
        "intent": majority_intent,
        "draft_reply": generic_reply,
        "escalate": False,
        "escalation_reason": "trivial_baseline_never_escalates",
    }


# Simple keyword rules — intentionally crude, hand-written from a skim of
# the data, NOT tuned against the golden set (that would make this baseline
# artificially strong and defeat its purpose as a floor).
KEYWORD_RULES = [
    ("account_access", [r"\blog\s?in\b", r"\bpassword\b", r"\blocked\b", r"\bprime membership\b"]),
    ("billing_dispute", [r"\bcharged?\b", r"\bcharge\b", r"\bbilled?\b", r"\bunauthorized\b"]),
    ("refund_or_return", [r"\brefund\b", r"\breturn\b"]),
    ("wrong_or_damaged_item", [r"\bwrong item\b", r"\bdamaged\b", r"\bbroken\b", r"\bdefective\b"]),
    ("late_or_missing_delivery", [r"\bnever arrived\b", r"\bmissing\b", r"\blate\b", r"\bnever showed\b"]),
    ("order_status", [r"\bwhere is my order\b", r"\btracking\b", r"\bstatus\b"]),
    ("product_question", [r"\bdoes (it|this)\b", r"\bwill this\b", r"\bcompatible\b"]),
]


def simple_baseline_classify(message: str) -> str:
    text = message.lower()
    for intent, patterns in KEYWORD_RULES:
        if any(re.search(p, text) for p in patterns):
            return intent
    return "general_complaint"  # fallback bucket


def simple_baseline_predict(message: str, retrieval_index, k: int = 1) -> dict:
    intent = simple_baseline_classify(message)
    precedents = retrieval_index.query(message, k=k)
    reply = precedents[0]["brand_text"] if precedents else \
        "Sorry to hear that. Please DM us your order details so we can help."
    top_sim = precedents[0]["similarity"] if precedents else 0.0

    # crude escalation proxy: escalate only if no precedent found at all
    escalate = top_sim < 0.2
    return {
        "intent": intent,
        "draft_reply": reply,
        "escalate": escalate,
        "escalation_reason": "simple_baseline_no_precedent" if escalate else "simple_baseline_has_precedent",
    }