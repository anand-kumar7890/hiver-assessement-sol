"""
Drafts a reply grounded in retrieved historical (customer -> brand reply)
precedent, rather than generating from the model's own general knowledge
of "how customer support sounds."
"""


def build_drafting_prompt(message: str, intent: str, precedents: list) -> list:
    precedent_block = "\n\n".join(
        f"Similar past customer message: {p['customer_text']}\n"
        f"How {'the brand'} actually replied: {p['brand_text']}"
        for p in precedents
    ) if precedents else "No close historical precedent found."

    system = (
        "You are drafting a reply as AmazonHelp, Amazon's customer support Twitter account. "
        "Match the brand's real tone and resolution style shown in the precedent examples below. "
        "Be concise (Twitter-length, under 280 characters), empathetic, and specific to this "
        f"customer's message. The classified intent is: {intent}.\n\n"
        f"--- Historical precedent for grounding ---\n{precedent_block}\n"
        "--- End precedent ---\n\n"
        "Do not invent order numbers, refund amounts, or policy details not implied by the "
        "customer's message or the precedent. If unsure, ask the customer to DM order details "
        "rather than guessing."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": message},
    ]


def draft_reply(message: str, intent: str, precedents: list) -> str:
    from llm_client import chat
    prompt = build_drafting_prompt(message, intent, precedents)
    reply = chat(prompt, temperature=0.4, max_tokens=400)
    return reply.strip() if reply else reply
