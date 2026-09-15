"""
Compares your hand-filled judge_agreement_scoring_sheet.csv against the
LLM judge's own scores (from results.json) to compute real agreement
metrics — this is the evidence that justifies trusting the judge at scale.
"""
import json
import csv
import sys

sys.path.insert(0, "../src")

with open("results.json", encoding="utf-8") as f:
    results = json.load(f)

judge_scores = results.get("llm_judge_scores_subset", [])

human_scores = []
with open("judge_agreement_scoring_sheet.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        human_scores.append(row)

if len(human_scores) != len(judge_scores):
    print(f"WARNING: row count mismatch — {len(human_scores)} human rows vs "
          f"{len(judge_scores)} judge rows. Using the shorter length.")

n = min(len(human_scores), len(judge_scores))

human_overall = [int(human_scores[i]["human_overall_1to5"]) for i in range(n)]
judge_overall = [judge_scores[i].get("overall", 0) for i in range(n)]

def agreement_report(human, judge, tolerance=1):
    assert len(human) == len(judge)
    n = len(human)
    exact = sum(1 for h, j in zip(human, judge) if h == j) / n
    within_tol = sum(1 for h, j in zip(human, judge) if abs(h - j) <= tolerance) / n
    mean_abs_diff = sum(abs(h - j) for h, j in zip(human, judge)) / n
    return {
        "n": n,
        "exact_agreement_rate": exact,
        f"agreement_within_{tolerance}_rate": within_tol,
        "mean_absolute_difference": mean_abs_diff,
    }

report = agreement_report(human_overall, judge_overall)

print("=== JUDGE-HUMAN AGREEMENT (overall score) ===")
print(f"N = {report['n']}")
print(f"Exact agreement rate: {report['exact_agreement_rate']:.1%}")
print(f"Agreement within 1 point: {report['agreement_within_1_rate']:.1%}")
print(f"Mean absolute difference: {report['mean_absolute_difference']:.2f}")
print()

diffs = [(human_scores[i]["example_id"], human_overall[i], judge_overall[i],
          abs(human_overall[i] - judge_overall[i])) for i in range(n)]
diffs.sort(key=lambda x: -x[3])
print("=== BIGGEST DISAGREEMENTS (for report discussion) ===")
for ex_id, h, j, d in diffs[:5]:
    if d > 0:
        print(f"[{ex_id}] human={h} judge={j} (diff={d})")

with open("judge_agreement_results.json", "w") as f:
    json.dump({
        "overall_agreement": report,
        "per_example": [
            {"example_id": human_scores[i]["example_id"],
             "human_overall": human_overall[i],
             "judge_overall": judge_overall[i]}
            for i in range(n)
        ],
    }, f, indent=2)
print()
print("Saved full results to judge_agreement_results.json")