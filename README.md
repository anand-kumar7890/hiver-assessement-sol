# AmazonHelp AI Support Agent — Hiver Take-Home

An AI support agent for **AmazonHelp (Twitter)** that classifies incoming customer messages into support intents, drafts a reply grounded in historically similar resolved cases, and decides whether the case can be auto-handled or should be escalated to a human — with an explicit reason for that decision.

## What the agent does

The pipeline has four main stages:

1. **Intent classification** — identifies the customer's support intent using an LLM and a 9-class taxonomy.
2. **Retrieval grounding** — retrieves historically similar customer/brand-reply pairs using sentence-transformer embeddings.
3. **Reply drafting** — generates a customer-facing draft grounded in the retrieved precedent.
4. **Escalation** — applies explicit, auditable rules to determine whether a human should handle the case.

Every message receives a draft reply. The escalation decision determines whether that draft should be acted on automatically or reviewed by a human.

---

## Repository structure

```text
src/
  data_loader.py       # Loads/reconstructs customer ↔ brand reply pairs
  retrieval.py         # Embedding-based nearest-neighbor retrieval
  intents.py           # Intent taxonomy + LLM classifier
  escalation.py        # Rule-based escalation decision
  reply_drafter.py     # LLM reply generation grounded in precedent
  agent.py             # Wires the pipeline into SupportAgent.handle()
  llm_client.py        # Groq API wrapper

cleaning/
  clean_data.py        # Dataset cleaning and preprocessing

eval/
  golden_set_schema.py             # Golden-set schema/template
  golden_set_labeled.csv            # Hand-labeled evaluation set
  golden_set_labeled_FINAL.csv      # Final hand-labeled evaluation set
  baselines.py                      # Trivial and simple non-LLM baselines
  metrics.py                        # Intent/escalation metrics + LLM judge
  run_eval.py                       # Main evaluation entrypoint
  Compute_judge_agreement.py        # Human vs. LLM judge agreement
  extract_failures.py               # Failure-analysis extraction
  make_scoring_sheet.py             # Creates judge scoring sheet
  results.json                      # Evaluation results
  results_checkpoint.jsonl           # Per-example resumable evaluation results
  judge_agreement_results.json      # Judge agreement results
  judge_agreement_scoring_sheet.csv # Human scoring sheet

data/
  make_mock_data.py                       # Generates synthetic data
  mock_twitter_support.csv                # Synthetic development data
  mock_twitter_support_with_debug_intent.csv # Synthetic data with debug labels

tests/
  test_agent_wiring.py       # Offline unit tests

report.md                    # Full report, results, failure analysis and decision log
requirements.txt             # Python dependencies
README.md                    # Project documentation
```

### Dataset files not included

The full Kaggle dataset and generated embedding caches are intentionally excluded from Git because of their size.

The expected local files are:

```text
data/twcs.csv
data/amazon_pairs_cleaned.csv
data/embedding_cache.npz
data/embedding_cache_cleaned.npz
```

These files are ignored through `.gitignore`.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/anand-kumar7890/hiver-assessement-sol.git
cd hiver-assessement-sol
```

### 2. Create and activate a virtual environment

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Groq API

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_api_key_here
```

Do **not** commit `.env` or API keys to Git.

The project uses the Groq API for LLM inference. A free Groq account can be used for development, subject to the provider's current limits.

### 5. Download the dataset

Download the **Customer Support on Twitter** dataset:

`thoughtvector/customer-support-on-twitter`

Place the raw CSV at:

```text
data/twcs.csv
```

The complete dataset is intentionally not committed to this repository because of its size.

---

## Running the tests

The wiring tests do not require a live API key.

```bash
python -m pytest tests/ -v
```

These tests verify core agent wiring and escalation behavior.

---

## Running the agent manually

Example:

```bash
python src/agent.py "My package never arrived and it's been 6 days!"
```

The agent returns the classification, confidence, retrieved precedent, generated reply, and escalation decision with its reason.

---

## Evaluation

The evaluation compares the LLM agent against two simple baselines:

* **Trivial baseline** — predicts the majority intent.
* **Simple baseline** — uses a non-LLM retrieval-based heuristic.
* **LLM agent** — performs intent classification, retrieval-grounded drafting, and rule-based escalation.

### Run the evaluation

After preparing the dataset and golden set:

```bash
python eval/run_eval.py \
    --golden_set eval/golden_set_labeled_FINAL.csv \
    --pairs_csv data/twcs.csv \
    --out eval/results.json
```

The evaluation produces:

* Intent accuracy
* Intent macro-F1
* Escalation precision
* Escalation recall
* Escalation F1
* Missed-escalation rate
* LLM-as-judge reply-quality scores

Results are written to:

```text
eval/results.json
```

The evaluation is checkpointed per example in:

```text
eval/results_checkpoint.jsonl
```

This allows an interrupted run to resume without unnecessarily repeating completed API calls.

---

## Current headline results

On the final hand-labeled golden set of **267 examples**:

| System           | Intent Accuracy | Intent Macro-F1 | Escalation F1 | Missed-Escalation Rate |
| ---------------- | --------------: | --------------: | ------------: | ---------------------: |
| Trivial baseline |           0.326 |           0.055 |         0.000 |                  0.210 |
| Simple baseline  |           0.333 |           0.244 |         0.000 |                  0.210 |
| LLM agent        |       **0.772** |       **0.663** |     **0.197** |              **0.184** |

The LLM agent substantially outperforms both baselines on intent classification.

However, escalation remains the main weakness: the agent caught only approximately **12.5% (7/56)** of examples that were labeled as requiring escalation. This means the escalation F1 score should not be interpreted as production-ready safety performance.

The full analysis and limitations are documented in `report.md`.

---

## Golden set

The final golden set contains **267 hand-labeled examples** sampled from a cleaned AmazonHelp dataset.

The sampling pipeline was:

```text
Raw AmazonHelp pairs
        ↓
Remove non-standalone fragments
        ↓
Remove non-English messages
        ↓
Cleaned pool of 111,570 pairs
        ↓
Random sample of 300
        ↓
Manual review/removal of unsuitable examples
        ↓
Final golden set of 267
```

The final escalation distribution is:

```text
TRUE  = 56  (~21%)
FALSE = 211 (~79%)
```

The intent distribution is intentionally not balanced. `late_or_missing_delivery` and `general_complaint` are the most common classes, while some classes have very few examples. Per-class metrics for those rare classes should therefore be interpreted cautiously.

---

## Judge-agreement check

Reply quality is evaluated using an LLM-as-judge.

The judge was separately compared with human scoring on **40 examples**:

* Exact agreement: **45.0%**
* Agreement within 1 point: **82.5%**
* Mean absolute difference: **0.72**
* Mean judge score: **3.75 / 5**

The agreement is sufficient for relative system comparisons, but not strong enough to use the judge as an unquestioned pass/fail gate for individual replies.

To reproduce the agreement analysis:

```python
from eval.metrics import judge_agreement_report

judge_agreement_report(
    your_human_overall_scores,
    judge_overall_scores_from_json
)
```

See `report.md` for the detailed interpretation.

---

## Failure analysis

The five main observed failure modes were:

### 1. Repeated-contact situations are not detected

Messages describing multiple previous failed contact attempts were sometimes auto-handled because the escalation rules did not explicitly detect repeated-contact language.

Examples include phrases such as:

```text
"repeatedly reported"
"my last 5 parcels"
"I have contacted you several times"
```

### 2. Financial-harm language is broader than the keyword list

Customers may describe billing harm using phrases such as:

```text
"charged twice"
"cheated"
"charging me $200 for..."
```

without using explicit keywords such as `fraud` or `unauthorized`.

### 3. Fraud-adjacent cases can be missed

A suspected fake seller may not contain obvious fraud keywords and can also be confused with ordinary damaged/wrong-item complaints.

### 4. Compound or ambiguous intents

Some messages genuinely combine multiple concerns, such as account/payment issues or complaints about pricing and billing.

### 5. Garbled or low-signal messages

Typos, abbreviations, and autocorrect-mangled messages can cause unstable classification even when the classifier reports high confidence.

The detailed examples and analysis are in `report.md`.

---

## What I would improve with one more week

Based directly on the observed failures, the highest-priority improvements would be:

1. Add a repeated-contact/pattern-of-failure escalation signal.
2. Expand or replace hard-coded escalation keywords using real customer examples.
3. Add an independent second labeler for part of the golden set.
4. Incorporate multi-turn conversation/thread context.
5. Fine-tune and compare a smaller classifier after collecting more labeled data.
6. Build regression tests from the observed failure cases and use active learning to prioritize new labels.

---

## Important limitations

The headline numbers should not be interpreted as production guarantees.

Key limitations include:

* The golden set contains only 267 examples across 9 intents.
* The intent distribution is highly uneven.
* The golden set was labeled by a single human labeler.
* Escalation labels involve judgment and are not objective ground truth.
* Approximately 28% of the original AmazonHelp pairs were removed during preprocessing before sampling.
* The evaluated data is a cleaned subsample rather than the complete production distribution.
* The classifier, escalation keyword layer, and reply drafter are English-only.
* The LLM judge and reply-generation model come from the same broad model family, which may introduce judge bias.
* Escalation recall is currently too low for production use in high-risk scenarios.

These limitations and the corresponding decision rationale are documented in `report.md`.

---

## Design decisions

Some important implementation decisions were made deliberately:

* **9-class intent taxonomy** to avoid fragmenting a relatively small golden set.
* **Retrieval grounding** instead of generating replies from the LLM without precedent.
* **Explicit rule-based escalation** because escalation is a trust/safety boundary that should be auditable and testable.
* **Human-in-the-loop reply sending** — the system drafts replies but does not automatically send them.
* **Word-boundary regex matching** for high-risk keywords to avoid false positives such as matching `sue` inside `issue`.
* **Sentence-transformers + brute-force cosine similarity** instead of introducing a vector database for the evaluation-scale corpus.
* **Checkpointed evaluation** so interrupted API runs can resume.
* **Model-specific handling** for different reasoning/JSON-mode requirements.

See `report.md` for the complete decision log.

---

## Synthetic demo data

The repository includes synthetic mock data generated by:

```bash
python data/make_mock_data.py
```

The mock files are intended for development and pipeline smoke tests only.

They are **not** used for the reported headline evaluation results.

---

## Citations / borrowed material

The project uses:

* **Customer Support on Twitter** dataset from Kaggle:
  `thoughtvector/customer-support-on-twitter`
* **sentence-transformers/all-MiniLM-L6-v2** for retrieval embeddings.
* **langdetect** for language detection.
* **Groq-hosted Qwen/gpt-oss open-weight models** for LLM inference and evaluation.

See `report.md` for the detailed decision log, model choices, and attribution.

---

## Author

**Anand Kumar**

Hiver Take-Home Assessment — AmazonHelp AI Support Agent
