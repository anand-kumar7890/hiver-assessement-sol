"""
Generates a small synthetic dataset that mimics the structure of the
Kaggle "Customer Support on Twitter" dataset, filtered to AmazonHelp-style
conversations. This is ONLY for local pipeline testing before the real
Kaggle CSV is available. It is not used for any reported metrics.

Real dataset columns (from Kaggle thoughtvector/customer-support-on-twitter):
tweet_id, author_id, inbound, created_at, text, response_tweet_id, in_response_to_tweet_id

We synthesize customer -> AmazonHelp reply pairs across intents that mirror
what actually shows up in the real data for this brand.
"""
import csv
import random
import uuid
from datetime import datetime, timedelta

random.seed(42)

# (intent_label, customer_message_templates, brand_reply_templates)
TEMPLATES = {
    "order_status": {
        "customer": [
            "Where is my order #{oid}? It's been 5 days.",
            "Hey @AmazonHelp any update on order {oid}? Tracking hasn't moved.",
            "My package for order {oid} says it shipped but tracking is stuck.",
        ],
        "brand": [
            "Hi, sorry for the delay! Please check your tracking link here: {link}. We're looking into the delay with the carrier.",
            "We understand the wait is frustrating. Your order {oid} is currently in transit, expected within 2 business days.",
        ],
    },
    "late_or_missing_delivery": {
        "customer": [
            "My order {oid} was supposed to arrive yesterday and never showed up!",
            "Delivery for {oid} marked as delivered but I never got it. This is unacceptable.",
            "@AmazonHelp package for {oid} is now 6 days late with no updates.",
        ],
        "brand": [
            "We're really sorry to hear that. Please DM us your order number so we can investigate and arrange a replacement or refund.",
            "Apologies for the inconvenience. We've flagged this with the carrier and will follow up with a resolution within 24 hours.",
        ],
    },
    "wrong_or_damaged_item": {
        "customer": [
            "I ordered a blender and got a damaged box with broken glass inside, order {oid}.",
            "Received the wrong item for order {oid}, I ordered headphones and got a phone case.",
            "Item {oid} arrived completely smashed, packaging was fine but the product is destroyed.",
        ],
        "brand": [
            "So sorry about that! Please reply with your order number and a photo via DM so we can process a replacement right away.",
            "That's not the experience we want for you. We'll get a free replacement or refund started once you confirm your order ID.",
        ],
    },
    "refund_or_return": {
        "customer": [
            "I returned order {oid} two weeks ago and still no refund.",
            "Requesting a refund for {oid}, item doesn't match the description at all.",
            "Can I get a return label for order {oid}? Wrong size.",
        ],
        "brand": [
            "We can help with that. Refunds typically take 3-5 business days once the return is received. Let us know your order ID via DM.",
            "You're welcome to start a return from Your Orders page, or DM us the order number and we'll generate a label for you.",
        ],
    },
    "billing_dispute": {
        "customer": [
            "I was charged twice for order {oid}, this is fraud, fix it now.",
            "There's an unauthorized charge on my card labeled Amazon, order {oid}, I never bought this.",
            "Why was I billed ${amt} more than the listed price for {oid}?",
        ],
        "brand": [
            "We take this seriously. Please DM your order number and the charge details so our billing team can investigate immediately.",
            "Sorry for the confusion. Can you send us the order ID via DM so we can review the charge and correct it if needed?",
        ],
    },
    "account_access": {
        "customer": [
            "I've been locked out of my Amazon account for 3 days, no email response.",
            "My Prime membership shows canceled but I never canceled it, need this fixed urgently.",
            "Can't log in, keeps saying my password is wrong even after reset.",
        ],
        "brand": [
            "We're sorry for the trouble. Please visit amazon.com/account-recovery or DM us so we can escalate this to our account security team.",
            "This needs a closer look from our account team. Please DM your registered email (not your password) so we can assist securely.",
        ],
    },
    "product_question": {
        "customer": [
            "Does the Echo Dot 5th gen work without wifi at all?",
            "Is the Kindle Paperwhite waterproof enough for the pool?",
            "Will this charger work with the new iPhone models?",
        ],
        "brand": [
            "Great question! You can find full specs on the product page, or feel free to DM us for more details.",
            "Thanks for asking! Compatibility details are listed under 'Product Information' on the listing page.",
        ],
    },
    "general_complaint": {
        "customer": [
            "Amazon customer service has gone downhill so much this year, so disappointed.",
            "Been a Prime member for 8 years and this is how you treat loyal customers?",
            "Absolutely fed up with how many issues I've had this month.",
        ],
        "brand": [
            "We're sorry to hear you're feeling this way. We'd like to make it right, please DM us more details.",
            "That's not the experience we want you to have. Let us know how we can help via DM.",
        ],
    },
    "spam_or_irrelevant": {
        "customer": [
            "@AmazonHelp check out my new mixtape link in bio!!",
            "lol @AmazonHelp remember when Alexa said something weird lmaooo",
            "@AmazonHelp unrelated but does anyone know a good pizza place downtown",
        ],
        "brand": [
            "",  # brand often doesn't reply to these
        ],
    },
}

INTENTS = list(TEMPLATES.keys())


def gen_row(tweet_id, author_id, text, inbound, in_response_to=None, response_id=None, ts=None):
    return {
        "tweet_id": tweet_id,
        "author_id": author_id,
        "inbound": inbound,
        "created_at": ts.strftime("%a %b %d %H:%M:%S +0000 %Y"),
        "text": text,
        "response_tweet_id": response_id if response_id is not None else "",
        "in_response_to_tweet_id": in_response_to if in_response_to is not None else "",
    }


def main(n_threads_per_intent=40, out_path="data/mock_twitter_support.csv"):
    rows = []
    tid = 1000
    base_time = datetime(2024, 1, 1)

    for intent in INTENTS:
        cust_templates = TEMPLATES[intent]["customer"]
        brand_templates = TEMPLATES[intent]["brand"]
        for i in range(n_threads_per_intent):
            oid = random.randint(100000, 999999)
            amt = random.choice([5, 10, 15, 20])
            cust_text = random.choice(cust_templates).format(oid=oid, amt=amt)
            brand_text = random.choice(brand_templates).format(
                oid=oid, link=f"amzn.to/track{oid}"
            )

            cust_tweet_id = tid
            tid += 1
            ts1 = base_time + timedelta(minutes=random.randint(0, 500000))

            if brand_text:
                brand_tweet_id = tid
                tid += 1
                ts2 = ts1 + timedelta(minutes=random.randint(2, 120))
                rows.append(
                    gen_row(
                        cust_tweet_id,
                        f"cust_{uuid.uuid4().hex[:8]}",
                        cust_text,
                        True,
                        response_id=brand_tweet_id,
                        ts=ts1,
                    )
                )
                rows.append(
                    gen_row(
                        brand_tweet_id,
                        "AmazonHelp",
                        brand_text,
                        False,
                        in_response_to=cust_tweet_id,
                        ts=ts2,
                    )
                )
            else:
                rows.append(
                    gen_row(
                        cust_tweet_id,
                        f"cust_{uuid.uuid4().hex[:8]}",
                        cust_text,
                        True,
                        ts=ts1,
                    )
                )

            # stash the ground-truth intent for our own reference only
            # (NOT part of the real dataset schema — written to a side file)
            rows[-1]["_debug_intent"] = intent if brand_text else "spam_or_irrelevant"
            if brand_text:
                rows[-2]["_debug_intent"] = intent

    fieldnames = [
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id",
    ]
    debug_fieldnames = fieldnames + ["_debug_intent"]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})

    debug_path = out_path.replace(".csv", "_with_debug_intent.csv")
    with open(debug_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=debug_fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} rows to {out_path}")
    print(f"Wrote debug version (with true intent labels) to {debug_path}")


if __name__ == "__main__":
    main()
