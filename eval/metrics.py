"""
Evaluation metrics against the golden set.

Three separate things are measured, deliberately kept separate rather than
collapsed into one "accuracy" number (see report section on misleading
headline numbers):
    1. Intent classification quality (per-class precision/recall/F1)
    2. Escalation decision quality (precision/recall against human judgment)
    3. Reply quality (LLM-as-judge rubric, with human-agreement check)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from collections import defaultdict


def intent_classification_report(predictions: list, golden: list) -> dict:
    """
    predictions, golden: lists of intent label strings, same order/length.
    Returns per-class precision/recall/F1 plus macro averages.
    Macro (not micro/accuracy) is the headline metric — accuracy alone would
    be dominated by whichever intent is most frequent in the golden set.
    """
    assert len(predictions) == len(golden)
    labels = sorted(set(golden) | set(predictions))

    per_class = {}
    for label in labels:
        tp = sum(1 for p, g in zip(predictions, golden) if p == label and g == label)
        fp = sum(1 for p, g in zip(predictions, golden) if p == label and g != label)
        fn = sum(1 for p, g in zip(predictions, golden) if p != label and g == label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        support = sum(1 for g in golden if g == label)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}

    accuracy = sum(1 for p, g in zip(predictions, golden) if p == g) / len(golden)
    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(per_class)

    return {"per_class": per_class, "accuracy": accuracy, "macro_f1": macro_f1}


def escalation_report(predicted_escalate: list, golden_escalate: list) -> dict:
    """
    Both lists of bools. Reports precision/recall/F1 for the "escalate" class
    specifically — recall matters more here (missing an escalation that
    should have happened is worse than an unnecessary one), so report both,
    don't just average them away.
    """
    assert len(predicted_escalate) == len(golden_escalate)
    tp = sum(1 for p, g in zip(predicted_escalate, golden_escalate) if p and g)
    fp = sum(1 for p, g in zip(predicted_escalate, golden_escalate) if p and not g)
    fn = sum(1 for p, g in zip(predicted_escalate, golden_escalate) if not p and g)
    tn = sum(1 for p, g in zip(predicted_escalate, golden_escalate) if not p and not g)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "missed_escalations_rate": fn / len(golden_escalate) if golden_escalate else 0.0,
    }


JUDGE_RUBRIC_PROMPT = """You are evaluating a draft customer support reply for quality.

Customer message: {customer_message}
Classified intent: {intent}
Historical precedent shown to the drafting model: {precedent}
Draft reply being evaluated: {draft_reply}

Score the draft reply on each dimension from 1 (poor) to 5 (excellent):
- grounded: Is the reply consistent with the historical precedent's approach/tone, without inventing details (order numbers, refund amounts, policies) not supported by the message or precedent?
- relevance: Does the reply actually address what the customer asked?
- tone: Is the tone empathetic and appropriate for a frustrated/concerned customer?
- actionability: Does the reply give the customer a clear next step?

Respond ONLY with JSON: {{"grounded": <1-5>, "relevance": <1-5>, "tone": <1-5>, "actionability": <1-5>, "overall": <1-5>, "rationale": "<one sentence>"}}
"""


def judge_reply(customer_message: str, intent: str, precedent: list, draft_reply: str) -> dict:
    from llm_client import chat_json, JUDGE_MODEL
    precedent_str = "; ".join(p["brand_text"] for p in precedent) if precedent else "none"
    prompt = [{"role": "user", "content": JUDGE_RUBRIC_PROMPT.format(
        customer_message=customer_message, intent=intent,
        precedent=precedent_str, draft_reply=draft_reply,
    )}]
    return chat_json(prompt, model=JUDGE_MODEL, temperature=0.0)


def judge_agreement_report(human_scores: list, judge_scores: list, tolerance: int = 1) -> dict:
    """
    Compares human-assigned 'overall' scores (1-5) to LLM judge 'overall'
    scores on the SAME subset of examples. Reports exact agreement,
    agreement within `tolerance` points, and mean absolute difference.
    This is the artifact that justifies trusting the judge at scale —
    run it on 30-50 examples minimum before trusting judge scores on the
    full golden set.
    """
    assert len(human_scores) == len(judge_scores)
    n = len(human_scores)
    exact = sum(1 for h, j in zip(human_scores, judge_scores) if h == j) / n
    within_tol = sum(1 for h, j in zip(human_scores, judge_scores) if abs(h - j) <= tolerance) / n
    mean_abs_diff = sum(abs(h - j) for h, j in zip(human_scores, judge_scores)) / n

    return {
        "n": n,
        "exact_agreement_rate": exact,
        f"agreement_within_{tolerance}_rate": within_tol,
        "mean_absolute_difference": mean_abs_diff,
    }
