"""
Main evaluation entrypoint. Runs the LLM agent AND both baselines against
the golden set, computes all metrics, and writes a results summary.

Usage:
    python run_eval.py --golden_set golden_set_labeled.csv \\
                        --pairs_csv ../data/mock_twitter_support.csv \\
                        --out results.json

Requires GROQ_API_KEY in the environment for the LLM agent and the judge
(the baselines run without it).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from data_loader import load_raw, build_pairs
from retrieval import RetrievalIndex
from agent import SupportAgent
from baselines import (
    trivial_baseline_fit, trivial_baseline_predict,
    simple_baseline_predict,
)
from metrics import (
    intent_classification_report, escalation_report,
    judge_reply, judge_agreement_report,
)


def parse_bool(x):
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in ("true", "1", "yes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden_set", required=True,
                         help="Path to hand-labeled golden set CSV (see golden_set_schema.py)")
    parser.add_argument("--pairs_csv", required=True,
                         help="Path to the raw Kaggle-format CSV to build the retrieval index from")
    parser.add_argument("--out", default="results.json")
    parser.add_argument("--skip_llm", action="store_true",
                         help="Skip the LLM agent + judge, run baselines only (no API key needed)")
    parser.add_argument("--judge_subset_n", type=int, default=40,
                         help="How many golden examples to also send through the LLM judge")
    args = parser.parse_args()

    golden = pd.read_csv(args.golden_set)
    golden["true_should_escalate"] = golden["true_should_escalate"].apply(parse_bool)

    # Auto-detect format: cleaned pairs (from cleaning/clean_data.py) have a
    # 'customer_text_clean' column already; raw Kaggle twcs.csv does not and
    # needs load_raw()+build_pairs() to reconstruct pairs first.
    header = pd.read_csv(args.pairs_csv, nrows=0).columns.tolist()
    is_cleaned = "customer_text_clean" in header
    # Cache filename matches agent.py's convention (embedding_cache_cleaned.npz
    # for cleaned data, embedding_cache.npz for raw) so both scripts share one
    # cache instead of each rebuilding it separately — rebuilding on 100k+
    # pairs takes over an hour, so this match matters.
    cache_name = "embedding_cache_cleaned.npz" if is_cleaned else "embedding_cache.npz"
    cache_path = os.path.join(os.path.dirname(args.pairs_csv) or ".", cache_name)

    if is_cleaned:
        print(f"[run_eval] Detected cleaned pairs format: {args.pairs_csv}")
        pairs = pd.read_csv(args.pairs_csv)
    else:
        print(f"[run_eval] Detected raw Kaggle format: {args.pairs_csv}; building pairs...")
        raw = load_raw(args.pairs_csv)
        pairs = build_pairs(raw)

    print(f"[run_eval] Retrieval index will use {len(pairs)} historical pairs.")
    index = RetrievalIndex(pairs, cache_path=cache_path)

    results = {"n_golden_examples": len(golden)}

    # --- Trivial baseline ---
    majority_intent = trivial_baseline_fit(golden["true_intent"].tolist())
    generic_reply = "Thanks for reaching out! Please DM us your order details so we can help further."
    trivial_intents, trivial_escalations = [], []
    for msg in golden["customer_message"]:
        r = trivial_baseline_predict(msg, majority_intent, generic_reply)
        trivial_intents.append(r["intent"])
        trivial_escalations.append(r["escalate"])

    results["trivial_baseline"] = {
        "intent": intent_classification_report(trivial_intents, golden["true_intent"].tolist()),
        "escalation": escalation_report(trivial_escalations, golden["true_should_escalate"].tolist()),
    }

    # --- Simple baseline ---
    simple_intents, simple_escalations = [], []
    for msg in golden["customer_message"]:
        r = simple_baseline_predict(msg, index)
        simple_intents.append(r["intent"])
        simple_escalations.append(r["escalate"])

    results["simple_baseline"] = {
        "intent": intent_classification_report(simple_intents, golden["true_intent"].tolist()),
        "escalation": escalation_report(simple_escalations, golden["true_should_escalate"].tolist()),
    }

    # --- LLM agent (requires GROQ_API_KEY) ---
    if not args.skip_llm:
        agent = SupportAgent(index)

        # Checkpointing: save each result to a JSONL file as soon as it's
        # computed, and on restart, skip any example_id already present in
        # that file. This means hitting a rate limit (or any crash) partway
        # through a long run never loses the work already done — rerunning
        # the exact same command just picks up where it left off instead of
        # burning quota redoing examples that already succeeded.
        checkpoint_path = os.path.splitext(args.out)[0] + "_checkpoint.jsonl"
        completed = {}
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    completed[row["example_id"]] = row
            print(f"[run_eval] Resuming: found {len(completed)} already-completed "
                  f"examples in {checkpoint_path}")

        checkpoint_f = open(checkpoint_path, "a", encoding="utf-8")
        agent_outputs_by_id = dict(completed)

        remaining = [
            (ex_id, msg) for ex_id, msg in zip(golden["example_id"], golden["customer_message"])
            if ex_id not in completed
        ]
        print(f"[run_eval] {len(remaining)} examples left to process "
              f"({len(completed)} already done).")

        for i, (ex_id, msg) in enumerate(remaining):
            out = agent.handle(msg)
            out["example_id"] = ex_id
            agent_outputs_by_id[ex_id] = out
            checkpoint_f.write(json.dumps(out, default=str) + "\n")
            checkpoint_f.flush()  # write immediately, don't lose progress on crash
            if (i + 1) % 10 == 0 or (i + 1) == len(remaining):
                print(f"[run_eval] Processed {i + 1}/{len(remaining)} remaining "
                      f"({len(agent_outputs_by_id)}/{len(golden)} total)")
        checkpoint_f.close()

        # Reassemble in the golden set's original order
        agent_outputs = [agent_outputs_by_id[ex_id] for ex_id in golden["example_id"]]
        agent_intents = [o["intent"] for o in agent_outputs]
        agent_escalations = [o["escalate"] for o in agent_outputs]

        results["llm_agent"] = {
            "intent": intent_classification_report(agent_intents, golden["true_intent"].tolist()),
            "escalation": escalation_report(agent_escalations, golden["true_should_escalate"].tolist()),
        }

        # LLM-as-judge on a subset
        judge_subset = agent_outputs[:args.judge_subset_n]
        judge_scores = []
        for out in judge_subset:
            score = judge_reply(out["cleaned_message"], out["intent"],
                                 out["retrieved_precedent"], out["draft_reply"])
            judge_scores.append(score)
        results["llm_judge_scores_subset"] = judge_scores
        results["llm_judge_mean_overall"] = (
            sum(s.get("overall", 0) for s in judge_scores) / len(judge_scores)
            if judge_scores else None
        )
        print(f"NOTE: Run compute_judge_agreement.py separately with your own "
              f"human scores on this same subset to get judge-agreement metrics.")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Wrote results to {args.out}")

    # Print a quick human-readable summary
    print("\n=== SUMMARY ===")
    for key in ("trivial_baseline", "simple_baseline", "llm_agent"):
        if key in results:
            acc = results[key]["intent"]["accuracy"]
            macro_f1 = results[key]["intent"]["macro_f1"]
            esc_f1 = results[key]["escalation"]["f1"]
            missed = results[key]["escalation"]["missed_escalations_rate"]
            print(f"{key:16s} | intent_acc={acc:.3f} macro_f1={macro_f1:.3f} "
                  f"escalation_f1={esc_f1:.3f} missed_escalation_rate={missed:.3f}")


if __name__ == "__main__":
    main()