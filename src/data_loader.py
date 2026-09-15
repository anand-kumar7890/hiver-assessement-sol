"""
Loads the Customer Support on Twitter dataset (Kaggle: thoughtvector/customer-support-on-twitter)
and reconstructs (customer_message -> brand_reply) pairs for a single brand.

Real dataset schema:
    tweet_id, author_id, inbound, created_at, text, response_tweet_id, in_response_to_tweet_id

`inbound` is True for customer tweets, False for brand tweets.
`author_id` for brand tweets is the brand's handle (e.g. "AmazonHelp").

We only need first-response pairs for grounding: a customer's inbound tweet
and the brand's first reply to it. Deeper multi-turn threads are collapsed
to (first customer message, first brand reply) for simplicity — see decision log.
"""
import re
import pandas as pd

DEFAULT_BRAND = "AmazonHelp"


def load_raw(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"tweet_id": str, "response_tweet_id": str,
                                       "in_response_to_tweet_id": str})
    # normalize inbound to bool if it came in as string
    if df["inbound"].dtype == object:
        df["inbound"] = df["inbound"].astype(str).str.lower().isin(["true", "1"])
    return df


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"@\w+", "", text)          # strip @mentions (incl. brand handle)
    text = re.sub(r"http\S+", "", text)        # strip URLs
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_pairs(df: pd.DataFrame, brand: str = DEFAULT_BRAND) -> pd.DataFrame:
    """
    Returns a DataFrame of reconstructed pairs:
        customer_tweet_id, customer_text, customer_text_clean,
        brand_tweet_id, brand_text, brand_text_clean, created_at
    Only pairs where the brand actually replied are included (unanswered
    customer tweets are kept separately for the "no precedent" / spam analysis).
    """
    df = df.copy()
    df["text"] = df["text"].fillna("")

    brand_tweets = df[(df["inbound"] == False) & (df["author_id"] == brand)]
    customer_tweets = df[df["inbound"] == True]

    # index customer tweets by tweet_id for fast lookup
    cust_by_id = customer_tweets.set_index("tweet_id")

    pairs = []
    for _, brand_row in brand_tweets.iterrows():
        parent_id = brand_row.get("in_response_to_tweet_id")
        if pd.isna(parent_id) or parent_id == "" or parent_id == "nan":
            continue
        if parent_id not in cust_by_id.index:
            continue
        cust_row = cust_by_id.loc[parent_id]
        if isinstance(cust_row, pd.DataFrame):  # duplicate ids, take first
            cust_row = cust_row.iloc[0]

        pairs.append({
            "customer_tweet_id": parent_id,
            "customer_text": cust_row["text"],
            "customer_text_clean": clean_text(cust_row["text"]),
            "brand_tweet_id": brand_row["tweet_id"],
            "brand_text": brand_row["text"],
            "brand_text_clean": clean_text(brand_row["text"]),
            "created_at": cust_row.get("created_at", None),
        })

    pairs_df = pd.DataFrame(pairs)
    pairs_df = pairs_df.drop_duplicates(subset=["customer_tweet_id"]).reset_index(drop=True)
    return pairs_df


def get_unanswered(df: pd.DataFrame, pairs_df: pd.DataFrame, brand: str = DEFAULT_BRAND) -> pd.DataFrame:
    """Customer tweets mentioning the brand that never got a reply from it —
    useful for spam/irrelevant analysis and for stress-testing 'no precedent found'."""
    customer_tweets = df[df["inbound"] == True].copy()
    answered_ids = set(pairs_df["customer_tweet_id"])
    unanswered = customer_tweets[~customer_tweets["tweet_id"].isin(answered_ids)]
    # rough filter: only ones that look like they mention the brand at all
    mask = unanswered["text"].str.contains(brand, case=False, na=False)
    return unanswered[mask]


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/mock_twitter_support.csv"
    raw = load_raw(path)
    pairs = build_pairs(raw)
    print(f"Loaded {len(raw)} raw rows, reconstructed {len(pairs)} customer->brand pairs")
    print(pairs.head(3).to_string())
