"""Small, dependency-free LLM client used by the attribution engine.

The default target is Ollama's native ``/api/generate`` endpoint (this is what
runs in the Colab notebook).  An OpenAI-compatible ``/v1/chat/completions``
endpoint is also supported so the same prompts can be tested against vLLM,
LM Studio, or a hosted provider without changing the engine.

Every response is cached on disk keyed by model + endpoint + prompt.  Colab
disconnects are common; the cache makes an analysis resumable and stops us
paying for the same prompt twice.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_MODEL = "llama3:8b"
DEFAULT_ENDPOINT = "http://localhost:11434/api/generate"
DEFAULT_CACHE = Path("runs") / "llm_cache"
REQUEST_TIMEOUT = 600
NUM_CTX = 8192


class LLMError(RuntimeError):
    """The model could not be reached or returned unusable JSON."""


@dataclass
class LLMUsage:
    calls: int = 0
    cache_hits: int = 0
    failures: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0

    def add(self, other: "LLMUsage") -> None:
        self.calls += other.calls
        self.cache_hits += other.cache_hits
        self.failures += other.failures
        self.prompt_tokens += other.prompt_tokens
        self.output_tokens += other.output_tokens

    def summary(self) -> Dict[str, int]:
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "failures": self.failures,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
        }


def _strip_json(text: Any) -> Dict[str, Any]:
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        raise LLMError("model returned non-text content")
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise LLMError("model did not return a JSON object")
        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError(f"model returned invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise LLMError("model returned JSON that is not an object")
    return data


def _endpoint_is_openai(endpoint: str, provider: Optional[str]) -> bool:
    if provider:
        return provider.lower() in {"openai", "openai-compatible", "chat"}
    low = endpoint.lower()
    return "/v1/" in low or low.rstrip("/").endswith("/chat/completions")


def _openai_url(endpoint: str) -> str:
    url = endpoint.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if "/v1" not in url:
        url += "/v1"
    return url + "/chat/completions"


class LLMClient:
    """Minimal JSON-only client with retries and an on-disk response cache."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        endpoint: str = DEFAULT_ENDPOINT,
        *,
        cache_dir: Optional[Path] = None,
        temperature: float = 0.1,
        num_ctx: int = NUM_CTX,
        timeout: int = REQUEST_TIMEOUT,
        retries: int = 2,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        self.model = model or DEFAULT_MODEL
        self.endpoint = endpoint or DEFAULT_ENDPOINT
        self.cache_dir = Path(cache_dir) if cache_dir else Path(
            os.environ.get("BOOK_TO_AUDIO_LLM_CACHE", DEFAULT_CACHE))
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.timeout = timeout
        self.retries = max(0, int(retries))
        self.provider = provider
        self.openai = _endpoint_is_openai(self.endpoint, provider)
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.verbose = verbose
        self.usage = LLMUsage()

    # ------------------------------------------------------------- reporting
    def summary(self) -> Dict[str, Any]:
        return {
            "provider": "openai-compatible" if self.openai else "ollama",
            "endpoint": self.endpoint,
            "model": self.model,
            **self.usage.summary(),
        }

    # ---------------------------------------------------------------- caching
    def _cache_path(self, prompt: str) -> Path:
        key = json.dumps({
            "provider": "openai" if self.openai else "ollama",
            "endpoint": self.endpoint,
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
        }, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / digest[:2] / f"{digest}.json"

    def _read_cache(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        data = payload.get("data")
        return data if isinstance(data, dict) else None

    def _write_cache(self, path: Path, data: Dict[str, Any],
                     usage: Dict[str, int]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"data": data, "usage": usage, "model": self.model},
                                      indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        except OSError:
            pass  # cache is an optimisation, not a requirement

    # ---------------------------------------------------------------- network
    def _request_ollama(self, prompt: str) -> tuple[Dict[str, Any], Dict[str, int]]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
            "think": False,
        }
        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        usage = {
            "prompt_tokens": int(body.get("prompt_eval_count") or 0),
            "output_tokens": int(body.get("eval_count") or 0),
        }
        return _strip_json(body.get("response", "")), usage

    def _request_openai(self, prompt: str) -> tuple[Dict[str, Any], Dict[str, int]]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Return only one JSON object. No prose, no markdown."},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            _openai_url(self.endpoint),
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        choices = body.get("choices") or []
        if not choices:
            raise LLMError("OpenAI-compatible endpoint returned no choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):  # some providers return content parts
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        u = body.get("usage") or {}
        usage = {
            "prompt_tokens": int(u.get("prompt_tokens") or 0),
            "output_tokens": int(u.get("completion_tokens") or 0),
        }
        return _strip_json(content or ""), usage

    # ----------------------------------------------------------------- public
    def query_json(self, prompt: str) -> Dict[str, Any]:
        """Return one JSON object, using the cache and retrying transient errors."""
        cache_path = self._cache_path(prompt)
        cached = self._read_cache(cache_path)
        if cached is not None:
            self.usage.cache_hits += 1
            return cached

        last: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                if self.openai:
                    data, usage = self._request_openai(prompt)
                else:
                    data, usage = self._request_ollama(prompt)
                self.usage.calls += 1
                self.usage.prompt_tokens += usage["prompt_tokens"]
                self.usage.output_tokens += usage["output_tokens"]
                self._write_cache(cache_path, data, usage)
                return data
            except (urllib.error.URLError, TimeoutError, OSError, ValueError, LLMError) as exc:
                last = exc
                if attempt < self.retries:
                    time.sleep(1.5 ** attempt)
                    continue
        self.usage.failures += 1
        raise LLMError(f"LLM request failed ({self.endpoint}, model={self.model}): {last}")
