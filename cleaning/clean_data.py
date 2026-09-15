"""
Cleans the reconstructed (customer -> brand reply) pairs before they're used
anywhere else in the pipeline — golden set sampling, retrieval grounding,
everything reads from the OUTPUT of this script, not the raw build_pairs()
output directly.

Two filters applied, both documented here because they're judgment calls
that belong in the decision log, not silent defaults buried in code:

1. LANGUAGE FILTER — keep English only.
   Why: the intent classifier prompt, escalation keyword list, and reply
   drafter are all written in English. Running them on non-English text
   would produce unpredictable, unverifiable results, and neither of us
   labeling the golden set can reliably judge intent in a language we
   don't read. This is a stated scope cut, not an oversight — see report.md.

2. FRAGMENT FILTER — remove messages that don't stand alone.
   Why: many rows in this dataset are the 2nd/3rd/4th message in a back-
   and-forth thread ("did all of those already", "UPS", "7", "Sold by
   Amazon") — they only make sense with prior thread context we're not
   modeling (see decision log: threads collapsed to first message only).
   Classifying these in isolation is closer to guessing than to intent
   classification. Filtered via: (a) a minimum word count, since almost
   all real fragments are very short, and (b) a small blocklist of common
   short acknowledgement/reply phrases seen in this dataset.

   This is a heuristic, not perfect — it will occasionally drop a genuine
   short complaint ("No parcel!") and occasionally keep an ambiguous one.
   That imprecision is itself worth a line in report.md's "what's
   misleading about my headline number" section: the golden set was
   sampled from pre-filtered data, not the raw distribution.
"""
import re
import sys
import pandas as pd

try:
    from langdetect import detect, detect_langs, DetectorFactory
    DetectorFactory.seed = 0  # deterministic results across runs
    _HAS_LANGDETECT = True
except ImportError:
    _HAS_LANGDETECT = False

MIN_WORD_COUNT = 4

# Short acknowledgement / reply-fragment phrases seen in this dataset that
# don't carry standalone support-request content. Matched as a whole
# lowercase string (after stripping punctuation), not a substring, to avoid
# accidentally dropping a longer message that happens to contain "thanks".
FRAGMENT_BLOCKLIST = {
    "ups", "usps", "fedex", "dhl", "sold by amazon", "helpful",
    "thanks", "thanks a lot", "thank you", "thank you so much",
    "yes", "no", "yep", "nope", "ok", "okay", "already done",
    "did all of those already", "updated feedback", "still no progress",
    "still waiting for the update", "haven't received any correspondence",
}


def _word_count(text: str) -> int:
    return len(text.split())


def _looks_like_fragment(text: str) -> bool:
    normalized = re.sub(r"[^\w\s]", "", text).strip().lower()
    if normalized in FRAGMENT_BLOCKLIST:
        return True
    if _word_count(text) < MIN_WORD_COUNT:
        return True
    return False


def _is_english(text: str) -> bool:
    if not _HAS_LANGDETECT:
        return True  # can't check — see note printed in main()
    # Strip long digit sequences (order IDs, tracking numbers) before
    # detection — langdetect gets confused by strings like
    # "403-5997631-1690750" and misclassifies otherwise-clear English
    # text as non-English. Found via testing against real dataset examples.
    text_for_detection = re.sub(r"\d[\d\-]{5,}\d", " ", text)
    if len(text_for_detection.strip()) < 3:
        return False
    try:
        # detect_langs() with a probability threshold, not detect()'s single
        # top guess: langdetect is a probabilistic model and its top-1 guess
        # flips unpredictably on short/ambiguous text (verified directly —
        # a genuine English sentence with a stripped-out order ID got
        # classified as Danish by detect() alone). Accepting English if it's
        # ABOVE a modest probability threshold, even if not the top guess,
        # is more robust for short customer-support messages than requiring
        # it to win outright.
        candidates = detect_langs(text_for_detection)
        for c in candidates:
            if c.lang == "en" and c.prob >= 0.3:
                return True
        return False
    except Exception:
        return False  # detection failure on garbage text -> treat as non-English


def clean_pairs(pairs_df: pd.DataFrame, text_col: str = "customer_text_clean",
                verbose: bool = True) -> pd.DataFrame:
    """
    Returns a filtered copy of pairs_df, keeping only English, non-fragment
    customer messages. Adds no new columns — same schema in, same schema out.
    """
    if not _HAS_LANGDETECT:
        print("[cleaning] WARNING: langdetect not installed — skipping language "
              "filter entirely. Install with: pip install langdetect")

    before = len(pairs_df)

    fragment_mask = pairs_df[text_col].apply(lambda t: not _looks_like_fragment(str(t)))
    after_fragment = pairs_df[fragment_mask]

    if _HAS_LANGDETECT:
        lang_mask = after_fragment[text_col].apply(lambda t: _is_english(str(t)))
        cleaned = after_fragment[lang_mask]
    else:
        cleaned = after_fragment

    if verbose:
        print(f"[cleaning] {before} pairs before cleaning")
        print(f"[cleaning] {len(after_fragment)} pairs after removing fragments "
              f"(<{MIN_WORD_COUNT} words or in blocklist)")
        print(f"[cleaning] {len(cleaned)} pairs after removing non-English "
              f"(removed {len(after_fragment) - len(cleaned)} more)")

    return cleaned.reset_index(drop=True)


if __name__ == "__main__":
    import os
    # Absolute path based on this file's location, not the current working
    # directory — works whether you run this from the project root
    # (`python cleaning/clean_data.py ...`) or from inside cleaning/.
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
    from data_loader import load_raw, build_pairs

    default_input = os.path.join(PROJECT_ROOT, "data", "mock_twitter_support.csv")
    default_output = os.path.join(PROJECT_ROOT, "data", "amazon_pairs_cleaned.csv")
    input_path = sys.argv[1] if len(sys.argv) > 1 else default_input
    output_path = sys.argv[2] if len(sys.argv) > 2 else default_output

    print(f"[cleaning] Loading raw data from {input_path}...")
    raw = load_raw(input_path)
    pairs = build_pairs(raw)
    cleaned = clean_pairs(pairs)
    cleaned.to_csv(output_path, index=False)
    print(f"[cleaning] Wrote {len(cleaned)} cleaned pairs to {output_path}")