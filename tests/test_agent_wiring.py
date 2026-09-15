"""
Tests the agent's orchestration logic (classification -> retrieval ->
escalation -> drafting) with the LLM calls mocked out, so this runs offline
and fast in CI without needing GROQ_API_KEY. Does NOT test model quality —
that's what the eval harness against the golden set is for.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unittest.mock import patch


def test_high_risk_keyword_forces_escalation():
    from escalation import decide_escalation
    result = decide_escalation(
        message="This is fraud, I never authorized this charge",
        intent="billing_dispute", confidence=0.95, retrieval_top_similarity=0.8,
    )
    assert result["escalate"] is True
    assert "high_risk_keyword" in result["reason"]


def test_account_access_always_escalates_even_with_high_confidence():
    from escalation import decide_escalation
    result = decide_escalation(
        message="I can't log into my account",
        intent="account_access", confidence=0.99, retrieval_top_similarity=0.9,
    )
    assert result["escalate"] is True
    assert "always_escalate_intent" in result["reason"]


def test_low_confidence_forces_escalation():
    from escalation import decide_escalation
    result = decide_escalation(
        message="something vague",
        intent="general_complaint", confidence=0.3, retrieval_top_similarity=0.8,
    )
    assert result["escalate"] is True
    assert "low_classifier_confidence" in result["reason"]


def test_no_precedent_forces_escalation():
    from escalation import decide_escalation
    result = decide_escalation(
        message="a totally novel issue",
        intent="order_status", confidence=0.9, retrieval_top_similarity=0.1,
    )
    assert result["escalate"] is True
    assert "no_close_precedent" in result["reason"]


def test_confident_intent_with_precedent_auto_handles():
    from escalation import decide_escalation
    result = decide_escalation(
        message="Where is my order?",
        intent="order_status", confidence=0.9, retrieval_top_similarity=0.8,
    )
    assert result["escalate"] is False


def test_full_agent_pipeline_with_mocked_llm():
    from data_loader import load_raw, build_pairs
    from retrieval import RetrievalIndex
    from agent import SupportAgent

    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "mock_twitter_support.csv")
    raw = load_raw(data_path)
    pairs = build_pairs(raw)
    index = RetrievalIndex(pairs)
    agent = SupportAgent(index)

    # agent.py does `from intents import classify_intent`, which binds the
    # name into agent's own namespace — patch it there, not on the source module.
    with patch("agent.classify_intent") as mock_classify, \
         patch("agent.draft_reply") as mock_draft:
        mock_classify.return_value = {
            "intent": "late_or_missing_delivery", "confidence": 0.9,
            "reasoning": "customer reports non-arrival",
        }
        mock_draft.return_value = "We're sorry for the delay! Please DM your order number."

        result = agent.handle("My package never arrived, it's been 6 days!")

        assert result["intent"] == "late_or_missing_delivery"
        assert result["escalate"] is False  # high confidence + precedent should exist
        assert result["draft_reply"].startswith("We're sorry")
        assert len(result["retrieved_precedent"]) > 0


if __name__ == "__main__":
    test_high_risk_keyword_forces_escalation()
    test_account_access_always_escalates_even_with_high_confidence()
    test_low_confidence_forces_escalation()
    test_no_precedent_forces_escalation()
    test_confident_intent_with_precedent_auto_handles()
    test_full_agent_pipeline_with_mocked_llm()
    print("All wiring tests passed.")
