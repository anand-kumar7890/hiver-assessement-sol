import json
import pandas as pd

golden = pd.read_csv('eval/golden_set_labeled.csv')
golden_lookup = golden.set_index('example_id').to_dict('index')

checkpoint = []
with open('eval/results_checkpoint.jsonl', encoding='utf-8') as f:
    for line in f:
        checkpoint.append(json.loads(line))

print('=== MISSED ESCALATIONS (true=TRUE, predicted=False) ===')
count = 0
for row in checkpoint:
    ex_id = row['example_id']
    truth = golden_lookup[ex_id]
    true_escalate = str(truth['true_should_escalate']).strip().upper() == 'TRUE'
    if true_escalate and not row['escalate']:
        count += 1
        print(f"[{ex_id}] intent={row['intent']} | true_reason={truth['true_escalation_reason']}")
        print(f"  MSG: {row['cleaned_message'][:150]}")
        print(f"  escalation_reason given: {row['escalation_reason']}")
        print()
        if count >= 8:
            break

print()
print('=== WRONG INTENT CLASSIFICATIONS (sample) ===')
count = 0
for row in checkpoint:
    ex_id = row['example_id']
    truth = golden_lookup[ex_id]
    if row['intent'] != truth['true_intent']:
        count += 1
        print(f"[{ex_id}] predicted={row['intent']} | true={truth['true_intent']}")
        print(f"  MSG: {row['cleaned_message'][:150]}")
        print()
        if count >= 8:
            break