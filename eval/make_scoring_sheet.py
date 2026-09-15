"""
Generates a blank scoring sheet for the same N examples the LLM judge
scored, so you can hand-score them WITHOUT seeing the judge's scores first
(seeing them first would anchor your own judgment).

After you fill this in, run compute_judge_agreement.py to compare.
"""
import json
import csv

with open("results.json", encoding="utf-8") as f:
    results = json.load(f)

judge_scores = results.get("llm_judge_scores_subset", [])
n = len(judge_scores)
print(f"Found {n} judge-scored examples.")

checkpoint = []
with open("results_checkpoint.jsonl", encoding="utf-8") as f:
    for line in f:
        checkpoint.append(json.loads(line))

subset = checkpoint[:n]

with open("judge_agreement_scoring_sheet.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "example_id", "customer_message", "draft_reply",
        "human_grounded_1to5", "human_relevance_1to5",
        "human_tone_1to5", "human_actionability_1to5", "human_overall_1to5",
    ])
    for row in subset:
        writer.writerow([
            row["example_id"], row["cleaned_message"], row["draft_reply"],
            "", "", "", "", "",
        ])

print(f"Wrote {len(subset)}-row scoring sheet to judge_agreement_scoring_sheet.csv")
print("Fill in the 5 human_* columns (1-5 each), then run compute_judge_agreement.py")