"""Thin OpenRouter client (OpenAI-compatible API) with disk cache, retries and cost tracking."""
from __future__ import annotations

import hashlib
import logging
import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

OPENROUTER_URL = "https://openrouter.ai/api/v1"
_ROOT = Path(__file__).resolve().parents[2]


# OpenRouter rejects requests (HTTP 402 "in_flight_budget_exhausted") when too many are in flight for a key with a small
# credit limit. Measured on this key: 2 concurrent requests are always fine, 4 mostly fail -> cap concurrency globally.
log = logging.getLogger("enron_synth.llm")
_SEM = threading.BoundedSemaphore(int(os.getenv("SYNTH_MAX_CONCURRENCY", "2")))


class BudgetExceeded(RuntimeError):
    pass


class CacheMiss(RuntimeError):
    """Raised in offline mode (SYNTH_OFFLINE=1) when a response is not in the disk cache."""


class LLM:
    """Usage:  LLM("openai/gpt-4.1-mini").chat([{"role":"user","content":"hi"}])

    * Every response is cached on disk (keyed by model+messages+params) so notebooks and
      experiments are reproducible and re-runs cost nothing.
    * `spent` accumulates the USD cost reported by OpenRouter; `max_cost` is a hard stop.
    """

    _lock = threading.Lock()
    spent = 0.0          # class-level: shared across all instances in the process
    calls = 0
    cache_hits = 0

    def __init__(self, model: str, api_key: Optional[str] = None, cache_dir: str | Path = _ROOT / "data" / "cache",
                 max_retries: int = 6, max_cost: float = 15.0, timeout: float = 60.0):
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("Set OPENROUTER_API_KEY (see .env.example).")
        self.model = model
        self.client = OpenAI(base_url=OPENROUTER_URL, api_key=key, timeout=timeout)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.max_cost = max_cost

    def _key(self, messages, temperature, max_tokens, extra) -> str:
        blob = json.dumps([self.model, messages, temperature, max_tokens, extra], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def chat(self, messages: list[dict], temperature: float = 0.7, max_tokens: int = 2500,
             use_cache: bool = True, cache_salt: str = "", **extra) -> str:
        """Return the assistant text. `cache_salt` lets callers force distinct samples for identical prompts."""
        ck = self._key(messages, temperature, max_tokens, [extra, cache_salt])
        path = self.cache_dir / f"{ck}.json"
        if use_cache and path.exists():
            with self._lock:
                LLM.cache_hits += 1
            return json.loads(path.read_text())["text"]

        if os.getenv("SYNTH_OFFLINE"):
            raise CacheMiss("SYNTH_OFFLINE=1 and no cached response for this prompt")
        if LLM.spent >= self.max_cost:
            raise BudgetExceeded(f"spent ${LLM.spent:.2f} >= cap ${self.max_cost:.2f}")

        last_err = None
        for attempt in range(self.max_retries):
            try:
                with _SEM:
                    r = self.client.chat.completions.create(
                        model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens,
                        extra_body={"usage": {"include": True}, **extra},
                    )
                text = r.choices[0].message.content or ""
                if not text.strip():
                    raise ValueError("empty completion")
                usage = getattr(r, "usage", None)
                cost = float(getattr(usage, "cost", 0.0) or 0.0) if usage else 0.0
                with self._lock:
                    LLM.spent += cost
                    LLM.calls += 1
                path.write_text(json.dumps({"text": text, "cost": cost, "model": self.model}))
                return text
            except Exception as e:  # noqa: BLE001 - network / rate limit / empty output
                last_err = e
                msg = str(e)
                log.warning("LLM call failed (model=%s attempt=%d): %s", self.model, attempt + 1, msg[:160])
                if "in_flight_budget" in msg or "429" in msg or "rate limit" in msg.lower():
                    time.sleep(5 + 10 * attempt)          # OpenRouter asks to wait for in-flight requests to settle
                    continue
                time.sleep(min(2 ** attempt, 20))
        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {last_err}")

    @classmethod
    def report(cls) -> str:
        return f"LLM calls={cls.calls} cache_hits={cls.cache_hits} spent=${cls.spent:.4f}"
