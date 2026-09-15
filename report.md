# Report: AmazonHelp AI Support Agent

**Status: TEMPLATE — sections marked `[FILL IN]` need your real numbers/examples
once you've run `eval/run_eval.py` against the real Kaggle data and your own
hand-labeled golden set. Everything else (framing, structure, methodology) is
ready to use as-is or adapt.**

---

## 1. Problem framing

**What "good" means for this agent, for this brand:**

For AmazonHelp specifically, "good" is not "correctly classify every tweet."
It's:
- Never auto-handle something that could cause real harm if wrong (account
  security, unauthorized charges) — recall on escalation for these categories
  matters more than overall accuracy.
- A drafted reply that's *safe to show a human reviewer as a starting point*
  even when it's imperfect — it should never invent order numbers, refund
  amounts, or policy commitments not grounded in the actual message or
  precedent.
- Correctly identifying non-support noise (spam/irrelevant mentions) so a
  human isn't wasting time triaging tweets that were never real support
  requests in the first place.

**What I chose not to build:**
- Multi-turn conversation state / follow-up tracking — the agent handles a
  single incoming message in isolation. A real deployment would need thread
  memory; out of scope for a 2-3 day take-home. [ADAPT/CONFIRM]
- Automatic sending of replies — the agent always produces a *draft*, even
  when "auto-handle" is decided; a human-in-the-loop send step is assumed to
  exist downstream. [ADAPT/CONFIRM]
- Fine-tuning a classifier — used few-shot prompting instead of fine-tuning a
  smaller supervised model on Banking77 or hand-labeled data, because the
  golden set (150-250 examples) is too small to fine-tune on responsibly, and
  few-shot prompting is more directly explainable in a live review. [ADAPT/CONFIRM]
- [FILL IN: anything else you deliberately cut]

---

## 2. Results vs. baselines

**[FILL IN — from `eval/results.json` after your real run]**

| System | Intent Accuracy | Intent Macro-F1 | Escalation Precision | Escalation Recall | Escalation F1 | Missed-Escalation Rate |
|---|---|---|---|---|---|---|
| Trivial baseline | | | | | | |
| Simple baseline | | | | | | |
| LLM agent | | | | | | |

**Reply quality (LLM-as-judge, mean `overall` score 1-5):**
- [FILL IN]

**Judge-human agreement (on N={40 or your chosen subset} examples):**
- Exact agreement rate: [FILL IN]
- Agreement within 1 point: [FILL IN]
- Mean absolute difference: [FILL IN]
- Interpretation: [FILL IN — e.g. "judge agrees within 1 point X% of the
  time, which is enough to trust it for relative comparisons across systems,
  but not for absolute pass/fail gating without spot-checks."]

---

## 3. Failure analysis — top 5 failure modes

For each: real example (customer message + agent output), what went wrong,
and your hypothesis for why.

### Failure mode 1: [FILL IN]
- Example: [FILL IN]
- What went wrong: [FILL IN]
- Hypothesis: [FILL IN]

### Failure mode 2: [FILL IN]
...

### Failure mode 3: [FILL IN]
...

### Failure mode 4: [FILL IN]
...

### Failure mode 5: [FILL IN]
...

**Likely candidates to look for, based on the pipeline's known weak points**
(check these first when reviewing your actual failures):
- Retrieval finding a *topically* similar but *situationally* wrong precedent
  (e.g. matching "late delivery" precedent for a message that's actually
  about a damaged item mentioned in passing).
- Classifier over-triggering `general_complaint` as a catch-all for messages
  that are actually a specific intent, just phrased angrily.
- Spam/irrelevant messages that use support-adjacent language ("my order of
  events for the day is chaotic lol @AmazonHelp") getting misclassified as
  real support requests.
- Escalation rule missing edge cases where high-risk language is implied but
  not keyword-matched (e.g. "this feels illegal" vs. the literal word
  "fraud").
- Reply drafter producing a reasonable-sounding but ungrounded specific
  promise (e.g. "your refund will arrive in 3 days") when the precedent
  didn't actually specify a timeline.

---

## 4. What is misleading about my headline number?

**This section is mandatory — be honest here, not defensive.**

Likely candidates for THIS project specifically:
- **Golden set size (150-250) is small relative to 9 intent classes** — some
  classes may have single-digit support in the golden set, making their
  per-class F1 noisy/unreliable even though macro-F1 looks like one clean
  number.
- **The golden set was sampled by me, from data I also built the system
  around** — sampling bias is possible even with a stated stratification
  method; I did not have an independent labeler to check my own labels.
- **The LLM judge and the reply-drafting model may share systematic biases**
  (e.g. both prefer verbose, hedge-y language) — judge-human agreement
  doesn't fully rule this out if my own human scoring was influenced by
  reading the draft in the same session as building the prompts.
- **Escalation ground truth is itself a judgment call I made**, not an
  objective label — reasonable people (or Hiver's own support team) might
  draw the escalation line differently, especially for `general_complaint`.
- **Evaluated on a subsample of a subsample** (Kaggle data itself is
  already a filtered/anonymized public release, and I evaluate a further
  subsample of AmazonHelp threads) — real-world message distribution and
  edge-case frequency may differ from what the golden set captures.
- [FILL IN: anything specific you notice once you see your real numbers —
  e.g. "macro-F1 looks fine but is propped up by one dominant easy class"]

---

## 5. What I'd do with one more week

- [FILL IN — likely candidates: expand golden set with a second independent
  labeler and compute inter-labeler agreement; add multi-turn thread context
  instead of single-message classification; fine-tune a small classifier on
  a larger labeled set and compare against the few-shot LLM approach; build
  an actual regression test suite of past failure examples; add active
  learning to prioritize which new messages most need human labels next.]

---

## Decision log

Non-obvious decisions and why, in the order they came up:

1. **Chose AmazonHelp over a smaller brand** — enough volume and intent
   diversity to fill a 200-example golden set without running out of real
   variety per intent.
2. **Collapsed multi-turn threads to (first customer message, first brand
   reply) pairs** for retrieval grounding — simpler and more tractable than
   full thread modeling for a 2-3 day project; documented as an explicit
   scope cut, not an oversight.
3. **Used an explicit rule table for escalation, not an LLM judgment call**
   — escalation is a trust/safety boundary; a rule table is auditable and
   testable in a way an LLM's escalate/don't-escalate call isn't.
4. **Kept the intent taxonomy to 9 classes**, derived from skimming real
   data rather than guessing upfront — a larger taxonomy would fragment an
   already-small golden set into unreliable per-class metrics.
5. **Used word-boundary regex for high-risk keyword matching, not substring
   matching** — caught during testing that substring matching on "sue"
   false-positives on ordinary words like "issue." (See
   `tests/test_agent_wiring.py::test_no_precedent_forces_escalation`, which
   is what surfaced this.)
6. **Chose Groq-hosted open-weight models (Llama 3.1/3.3) over a proprietary
   API** — fast/cheap enough for iteration within the time budget, and
   staying open-weight keeps the "you may use any LLM API or open model"
   framing honest even though inference is hosted, not local.
7. **Used a stronger model (Llama-3.3-70b) as the judge than the model doing
   classification/generation (Llama-3.1-8b)** — a same-or-weaker judge model
   risks systematically favoring its own generation patterns.
8. **Built the retrieval index with sentence-transformers + brute-force
   cosine similarity, no vector DB** — at subsample scale (thousands, not
   millions, of pairs), a vector DB adds complexity with no real latency or
   recall benefit, and is harder to explain/debug live.
9. **Every message gets a draft reply, even ones flagged for escalation** —
   a human reviewer benefits from a starting point rather than a blank page;
   changed the framing from "agent replies or doesn't" to "agent always
   drafts, routing decides who acts on it."
10. **Reported macro-F1 as the headline classification metric, not raw
    accuracy** — raw accuracy would be dominated by whichever intent is most
    frequent in the golden set and would hide poor performance on rarer but
    important intents like `account_access`.
11. **Reported escalation recall and precision separately, not just F1** —
    missing an escalation that should have happened (false negative) is a
    materially different kind of error than an unnecessary escalation
    (false positive), and collapsing them into one F1 number obscures which
    kind of error the system is actually making.
12. **[FILL IN based on your real run — e.g. any threshold you tuned, why
    a particular confidence cutoff, any data cleaning quirk you hit]**
13. **[FILL IN]**
14. **[FILL IN]**
15. **[FILL IN]**

---

## Golden set sampling & labeling note

**[FILL IN once you've built the real golden set — describe:]**
- How many examples per intent (stratified? proportional to natural
  frequency? oversampled rare/edge cases?)
- Any deliberate inclusion of hard cases (ambiguous intent, borderline
  escalation calls, spam that looks like real complaints)
- Who labeled it (just you? note if so — see "misleading headline number"
  section above)
- Any examples you excluded and why (e.g. non-English tweets, threads with
  missing brand replies)

---

## Citations / borrowed material

- Dataset: Kaggle `thoughtvector/customer-support-on-twitter`
  (https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
- Retrieval embedding model: `sentence-transformers/all-MiniLM-L6-v2`
  (Reimers & Gurevych, sentence-transformers library)
- LLM inference: Groq API serving Meta's Llama 3.1/3.3 open-weight models
- [FILL IN: any code patterns, Stack Overflow answers, or AI-assistant
  suggestions you incorporated substantially, per the assignment's citation
  rule]
