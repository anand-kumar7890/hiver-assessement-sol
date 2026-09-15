"""
Schema for the hand-labeled golden evaluation set (150-250 examples).

Each row is ONE customer message with human-assigned ground truth. Written
by a person (you), not generated — see decision_log.md for sampling method.

Columns:
    example_id           : str, unique id
    customer_message     : str, raw text as it appeared (post-cleaning is fine)
    true_intent           : str, one of intents.INTENTS keys, assigned by you
    true_should_escalate  : bool, your judgment of whether a human should
                             handle this (not the model's escalation output)
    true_escalation_reason: str, brief free-text reason for your call
    reference_reply       : str, OPTIONAL — a reply you consider good/acceptable,
                             used as an anchor for the LLM judge's reply-quality
                             scoring. Can be adapted from the brand's actual
                             historical reply to a similar message, or written
                             fresh if no good precedent exists.
    difficulty_note       : str, OPTIONAL — flag ambiguous/edge-case examples
                             (useful later for failure analysis and for
                             explaining disagreement with the LLM judge)
    sampling_stratum      : str, which stratum this was drawn from (e.g.
                             "late_delivery_random_sample", "spam_stress_test",
                             "ambiguous_multi_intent") — required for the
                             "how did you sample" writeup
"""
import csv

GOLDEN_SET_COLUMNS = [
    "example_id",
    "customer_message",
    "true_intent",
    "true_should_escalate",
    "true_escalation_reason",
    "reference_reply",
    "difficulty_note",
    "sampling_stratum",
]


def write_template(out_path: str, candidate_messages: list):
    """
    Given a list of candidate raw customer messages (e.g. pulled from
    build_pairs() or get_unanswered()), writes a CSV template with empty
    label columns ready for hand-labeling in a spreadsheet.
    """
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=GOLDEN_SET_COLUMNS)
        writer.writeheader()
        for i, msg in enumerate(candidate_messages):
            writer.writerow({
                "example_id": f"ex_{i:04d}",
                "customer_message": msg,
                "true_intent": "",
                "true_should_escalate": "",
                "true_escalation_reason": "",
                "reference_reply": "",
                "difficulty_note": "",
                "sampling_stratum": "",
            })
    print(f"Wrote {len(candidate_messages)}-row labeling template to {out_path}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "../src")
    from data_loader import load_raw, build_pairs

    raw = load_raw("../data/mock_twitter_support.csv")
    pairs = build_pairs(raw)

    # Simple stratified-ish sample for demonstration: take a slice across
    # the dataframe (already roughly balanced across intents in mock data).
    # For the REAL run: stratify explicitly per true intent bucket, oversample
    # rare intents and edge cases, and randomly sample within each stratum.
    sample = pairs["customer_text_clean"].sample(n=min(60, len(pairs)), random_state=42).tolist()
    write_template("golden_set_template.csv", sample)
