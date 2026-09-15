"""
Intent taxonomy for AmazonHelp, derived from manually skimming a sample of
the real dataset (see decision_log.md for sampling notes). Kept deliberately
small — 9 intents — because a large taxonomy fragments the golden set and
makes per-class metrics unreliable at this sample size.
"""

INTENTS = {
    "order_status": "Customer asking for a status/tracking update on an order that hasn't necessarily gone wrong yet.",
    "late_or_missing_delivery": "Order is confirmed late, marked delivered but not received, or significantly overdue.",
    "wrong_or_damaged_item": "Customer received the wrong item, or the item arrived damaged/defective.",
    "refund_or_return": "Customer wants a refund, return, or is following up on a refund/return already in progress.",
    "billing_dispute": "Customer disputes a charge: double charge, unauthorized charge, wrong amount billed.",
    "account_access": "Login issues, locked account, password reset problems, unexpected Prime/membership status change.",
    "product_question": "Pre-purchase or general product question with no specific order/issue to resolve.",
    "general_complaint": "Frustration/complaint about service quality with no single specific actionable request.",
    "spam_or_irrelevant": "Not a genuine support request — jokes, unrelated content, or spam mentioning the brand.",
}

# Intents that should NEVER be auto-handled regardless of classifier confidence.
ALWAYS_ESCALATE_INTENTS = {"account_access"}

# Keyword signals that force escalation regardless of intent (safety net,
# not a substitute for the classifier — see decision log on why both exist).
HIGH_RISK_KEYWORDS = [
    "fraud", "unauthorized", "lawyer", "legal action", "sue", "bbb",
    "better business bureau", "identity theft", "hacked",
]

CONFIDENCE_ESCALATION_THRESHOLD = 0.6


def build_classification_prompt(message: str) -> list:
    intent_list = "\n".join(f"- {k}: {v}" for k, v in INTENTS.items())
    system = (
        "You are an intent classifier for Amazon customer support tweets. "
        "Classify the customer's message into exactly one of the following intents:\n"
        f"{intent_list}\n\n"
        "Respond ONLY with a JSON object: "
        '{"intent": "<one of the intent keys above>", "confidence": <float 0-1>, '
        '"reasoning": "<one short sentence>"}'
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": message},
    ]


def classify_intent(message: str) -> dict:
    """Returns {"intent": str, "confidence": float, "reasoning": str}"""
    from llm_client import chat_json
    prompt = build_classification_prompt(message)
    result = chat_json(prompt)
    if result.get("intent") not in INTENTS:
        result["intent"] = "general_complaint"  # safe fallback, will trigger review
        result["confidence"] = min(result.get("confidence", 0.5), 0.4)
    return result
