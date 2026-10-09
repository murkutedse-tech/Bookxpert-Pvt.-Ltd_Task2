"""Step 6 - Answer generation. One small function per LLM provider.

Choose the provider with LLM_PROVIDER in .env:
  ollama    - free, local, no API key (default). Needs Ollama running.
  gemini    - Google Gemini, needs GEMINI_API_KEY (free tier available)
  openai    - OpenAI GPT, needs OPENAI_API_KEY
  anthropic - Anthropic Claude, needs ANTHROPIC_API_KEY
  none      - no LLM; the bot returns the best matching passage instead
API keys are only ever read from environment variables.
"""

from __future__ import annotations

import os

import requests

from .config import Settings


class LLMError(RuntimeError):
    pass


def _require_key(name: str) -> str:
    key = os.getenv(name, "").strip()
    if not key:
        raise LLMError(f"{name} is not set. Add it to your .env file (see .env.example).")
    return key


def _ollama(system: str, user: str, s: Settings) -> str:
    try:
        r = requests.post(
            f"{s.ollama_host.rstrip('/')}/api/chat",
            json={
                "model": s.ollama_model,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=s.request_timeout,
        )
    except requests.ConnectionError as exc:
        raise LLMError(f"Could not reach Ollama at {s.ollama_host}. Is `ollama serve` running?") from exc
    if r.status_code == 404:
        raise LLMError(f"Ollama model '{s.ollama_model}' not found. Run `ollama pull {s.ollama_model}`.")
    r.raise_for_status()
    return r.json()["message"]["content"]


def _gemini(system: str, user: str, s: Settings) -> str:
    key = _require_key("GEMINI_API_KEY")
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{s.gemini_model}:generateContent",
        headers={"x-goog-api-key": key},
        json={
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0},
        },
        timeout=s.request_timeout,
    )
    if not r.ok:
        raise LLMError(f"Gemini API error {r.status_code}: {r.text[:300]}")
    parts = r.json()["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts)


def _openai(system: str, user: str, s: Settings) -> str:
    key = _require_key("OPENAI_API_KEY")
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": s.openai_model,
            "temperature": 0,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        },
        timeout=s.request_timeout,
    )
    if not r.ok:
        raise LLMError(f"OpenAI API error {r.status_code}: {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"]


def _anthropic(system: str, user: str, s: Settings) -> str:
    _require_key("ANTHROPIC_API_KEY")
    try:
        import anthropic
    except ImportError as exc:
        raise LLMError("Install the Anthropic SDK: pip install anthropic") from exc
    client = anthropic.Anthropic(timeout=s.request_timeout)
    message = client.messages.create(
        model=s.anthropic_model,
        max_tokens=16000,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(block.text for block in message.content if block.type == "text")


PROVIDERS = {"ollama": _ollama, "gemini": _gemini, "openai": _openai, "anthropic": _anthropic}


def generate(system: str, user: str, settings: Settings) -> str:
    provider = settings.llm_provider
    if provider not in PROVIDERS:
        raise LLMError(f"Unknown LLM_PROVIDER '{provider}'. Use one of: {', '.join(PROVIDERS)}, none")
    return PROVIDERS[provider](system, user, settings).strip()
