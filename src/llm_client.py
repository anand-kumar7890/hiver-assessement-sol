"""
Thin wrapper around Groq's OpenAI-compatible API.

Requires GROQ_API_KEY in the environment (see .env.example).
Uses:
    - CLASSIFY_MODEL for intent classification and reply drafting (fast, cheap)
    - JUDGE_MODEL for LLM-as-judge scoring (stronger, to avoid a weak model
      grading its own homework more favorably than it should)

Get a free key at https://console.groq.com/keys
"""
import json
import os
import time

# Automatically load variables from a .env file in the project root, if present.
# This means you never have to manually export GROQ_API_KEY in your shell —
# just keep it in .env and this loads it every time the script runs.
from dotenv import load_dotenv
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

CLASSIFY_MODEL = os.environ.get("CLASSIFY_MODEL", "openai/gpt-oss-20b")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "openai/gpt-oss-120b")

# Both the gpt-oss and qwen3 model families on Groq are REASONING models:
# before writing the visible answer, they spend part of the token budget on
# hidden internal "thinking" tokens. The two families are controlled
# differently, though — see decision log:
#   - gpt-oss-20b/120b: reasoning_effort accepts 'low'/'medium'/'high'.
#     If max_tokens is too small, the whole budget can be consumed by
#     reasoning, leaving an EMPTY visible response.
#   - qwen3.x models: reasoning_effort only accepts 'none'/'default' (NOT
#     'low'/'medium'/'high' — passing those causes a 400 error). More
#     importantly, when JSON mode (response_format=json_object) is requested,
#     Qwen's default reasoning_format ('raw') embeds literal <think>...</think>
#     tags INSIDE the response content, which breaks JSON parsing entirely
#     (Groq itself rejects it with 'Failed to validate JSON'). Setting
#     reasoning_format='hidden' strips the thinking tags so only clean JSON
#     comes back. This was found by hitting the actual 400 error mid-eval-run.
REASONING_EFFORT = os.environ.get("REASONING_EFFORT", "low")

_client = None


def _get_client():
    global _client
    if _client is None:
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Copy .env.example to .env, add your key "
                "from https://console.groq.com/keys, and `source .env` (or use "
                "python-dotenv) before running."
            )
        _client = Groq(api_key=api_key)
    return _client


def chat(messages, model: str = CLASSIFY_MODEL, temperature: float = 0.2,
         max_tokens: int = 512, json_mode: bool = False, retries: int = 3,
         reasoning_effort: str = REASONING_EFFORT):
    """Thin chat completion wrapper with basic retry on transient errors.

    Also retries with a larger max_tokens if a reasoning model returns empty
    content because it spent the whole budget on hidden thinking — rather
    than silently returning "" and letting that flow downstream as a blank
    reply, which is what happened before this fix.
    """
    client = _get_client()
    kwargs = dict(model=model, messages=messages, temperature=temperature,
                  max_tokens=max_tokens)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    is_gpt_oss = "gpt-oss" in model
    is_qwen3 = "qwen3" in model or "qwen/qwen" in model

    if is_gpt_oss:
        kwargs["reasoning_effort"] = reasoning_effort
    elif is_qwen3:
        # qwen3 family only accepts none/default, never low/medium/high
        kwargs["reasoning_effort"] = "none"
        if json_mode:
            # Prevents <think>...</think> tags from contaminating JSON output
            kwargs["reasoning_format"] = "hidden"

    last_err = None
    current_max_tokens = max_tokens
    for attempt in range(retries):
        try:
            kwargs["max_tokens"] = current_max_tokens
            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content
            finish_reason = resp.choices[0].finish_reason
            if not content and finish_reason == "length":
                # Ran out of tokens mid-reasoning before any visible content —
                # double the budget and retry rather than returning "".
                current_max_tokens = current_max_tokens * 2
                print(f"[llm_client] Empty content, likely reasoning-token "
                      f"exhaustion (finish_reason=length). Retrying with "
                      f"max_tokens={current_max_tokens}.")
                continue
            return content
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    if last_err:
        raise last_err
    raise RuntimeError(
        f"Model '{model}' returned empty content after {retries} attempts "
        f"even after raising max_tokens to {current_max_tokens}. Check the "
        f"prompt isn't triggering excessive reasoning, or raise max_tokens "
        f"further."
    )


def chat_json(messages, model: str = CLASSIFY_MODEL, temperature: float = 0.1,
              max_tokens: int = 512) -> dict:
    """Chat completion that expects and parses a JSON object response."""
    raw = chat(messages, model=model, temperature=temperature,
               max_tokens=max_tokens, json_mode=True)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # occasionally models wrap JSON in markdown fences despite json_mode
        cleaned = raw.strip().strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
        return json.loads(cleaned)