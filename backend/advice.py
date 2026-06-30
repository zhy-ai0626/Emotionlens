"""EmotionLens · L4 Speech Coach — LLM advice generator.

Reads API key from env var EMO_LLM_API_KEY.
Reads base_url / model from config.local.json (at project root).
Falls back to rule-based English advice when key is missing or call fails.
NEVER logs or prints the key.
"""
import os
import json
import httpx
from backend.config import rule_based_advice


def _load_llm_config() -> dict:
    """Load base_url and model from config.local.json. Returns {} on failure."""
    # Try several possible locations for config.local.json
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "..", "config.local.json"),
        os.path.join(os.path.dirname(__file__), "..", "config.local.json"),
        "config.local.json",
    ]
    for path in candidates:
        try:
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            continue
    return {}


def gen_advice(metrics: dict) -> tuple[str, str]:
    """Generate coaching advice. Returns (advice_text, source) where source is "llm" or "rule".

    API key sources (in priority order):
      1. Env var EMO_LLM_API_KEY
      2. config.local.json → "api_key" field (gitignored, for local dev convenience)
    """
    cfg = _load_llm_config()
    key = os.getenv("EMO_LLM_API_KEY") or cfg.get("api_key", "")
    if not key:
        return rule_based_advice(metrics), "rule"
    base_url = cfg.get("base_url", "")
    model = cfg.get("model", "")
    if not base_url or not model:
        return rule_based_advice(metrics), "rule"

    system_prompt = (
        "You are a speech coach. Given the metrics, reply in ENGLISH with 3-4 "
        "specific, encouraging improvement tips. No greetings, no preamble. "
        "Keep each tip to one sentence."
    )
    user_prompt = f"Speech metrics: {json.dumps(metrics, ensure_ascii=False)}"

    try:
        r = httpx.post(
            f"{base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.7,
                "max_tokens": 200,
            },
            timeout=15.0,
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()
        return text, "llm"
    except Exception:
        return rule_based_advice(metrics), "rule"
