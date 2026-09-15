"""
Escalation decision logic.

Deliberately implemented as an explicit, inspectable rule table rather than
an LLM judgment call — escalation is a trust/safety boundary, and a rule
table is auditable, testable, and directly reviewable in the report/decision
log. The classifier's job is to categorize; this module's job is to decide
who's allowed to act on that categorization.
"""
import re

from intents import ALWAYS_ESCALATE_INTENTS, HIGH_RISK_KEYWORDS, CONFIDENCE_ESCALATION_THRESHOLD


def decide_escalation(message: str, intent: str, confidence: float,
                       retrieval_top_similarity: float,
                       min_precedent_similarity: float = 0.35) -> dict:
    """
    Returns {"escalate": bool, "reason": str}
    Checked in priority order; first match wins.

    Keyword matching uses word boundaries (\\b) deliberately: a naive
    substring check on "sue" would false-positive on "issue", "pursue",
    "tissue", etc. Caught by test_no_precedent_forces_escalation — see
    decision log.
    """
    text_lower = message.lower()

    for kw in HIGH_RISK_KEYWORDS:
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, text_lower):
            return {"escalate": True, "reason": f"high_risk_keyword:'{kw}'"}

    if intent in ALWAYS_ESCALATE_INTENTS:
        return {"escalate": True, "reason": f"always_escalate_intent:{intent}"}

    if confidence < CONFIDENCE_ESCALATION_THRESHOLD:
        return {"escalate": True,
                "reason": f"low_classifier_confidence:{confidence:.2f}<{CONFIDENCE_ESCALATION_THRESHOLD}"}

    if retrieval_top_similarity < min_precedent_similarity:
        return {"escalate": True,
                "reason": f"no_close_precedent:{retrieval_top_similarity:.2f}<{min_precedent_similarity}"}

    return {"escalate": False, "reason": "auto_handle:confident_intent_with_precedent"}
