import os
import httpx
import logging
import json
import asyncio
import threading
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
from app.core.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    DATA_DIR,
    LMSTUDIO_URL,
    LMSTUDIO_MODEL,
    PROVIDER,
)
from app.core.logging_config import configure_logging
from app.core.tracing import start_span
import time
from app.core.messages import LLMMessage, to_openrouter_messages

# Optional official OpenRouter SDK (preferred when installed)
try:  # pragma: no cover - depends on optional dependency
    from openrouter import OpenRouter
    _HAS_SDK = True
except Exception:  # pragma: no cover - best-effort import
    OpenRouter = None
    _HAS_SDK = False

configure_logging()
logger = logging.getLogger(__name__)
OFFLINE_QUEUE_ENABLED = os.getenv("SEARCHONE_LLM_OFFLINE_QUEUE") in ("1", "true", "yes", "on")
OFFLINE_QUEUE_FILE = DATA_DIR / "llm_offline_queue.jsonl"
LOCAL_ONLY_MODEL = "local-fallback"
LLM_CONCURRENCY = int(os.getenv("SEARCHONE_LLM_CONCURRENCY", "3")) if os.getenv("SEARCHONE_LLM_CONCURRENCY", "").strip() != "" else 3
# Avoid global async semaphores bound to a stale event loop; create per-client
_sync_semaphore = threading.BoundedSemaphore(LLM_CONCURRENCY) if LLM_CONCURRENCY > 0 else None

class LLMClient:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, use_sdk: bool = True, fallback_local: bool = False):
        self.api_key = api_key or OPENROUTER_API_KEY
        self.model = model or OPENROUTER_MODEL
        self.lmstudio_url = LMSTUDIO_URL
        self.lmstudio_model = LMSTUDIO_MODEL
        self.provider = (PROVIDER or "openrouter").lower()
        # optional OpenRouter metadata headers (only used for httpx fallback)
        self.site = os.getenv("OPENROUTER_SITE")
        self.title = os.getenv("OPENROUTER_TITLE", "SearchOne")
        self.use_sdk = bool(use_sdk and _HAS_SDK and self.api_key)
        # httpx fallback automatically uses official host unless overridden
        self.http_fallback_url = os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions")
        # Local fallback is disabled by default; force explicit opt-in via parameter.
        self.fallback_local = bool(fallback_local)
        # per-client async semaphore to avoid cross-loop binding
        self._async_semaphore = asyncio.Semaphore(LLM_CONCURRENCY) if LLM_CONCURRENCY > 0 else None

    async def generate(self, prompt: str = "", *, messages: Optional[List[Union[LLMMessage, Dict[str, Any]]]] = None, max_tokens: int = 512, tools: Optional[list] = None, tool_choice: Optional[str] = None) -> dict:
        """Generate text using OpenRouter (prefers SDK when available). Returns dict with text and raw payload."""
        sem = self._async_semaphore
        if sem:
            await sem.acquire()
        if self.fallback_local:
            if sem:
                sem.release()
            return {"text": self._local_answer(prompt), "raw": None, "tool_calls": []}
        # 1) Prefer LM Studio local server when provider=local
        if self.provider == "local":
            if not self.lmstudio_url:
                if sem:
                    sem.release()
                raise RuntimeError("PROVIDER=local mais LMSTUDIO_URL manquant.")
            try:
                resp = await self._generate_lmstudio_async(messages or [{"role": "user", "content": prompt}], max_tokens, tools, tool_choice)
                return resp
            finally:
                if sem:
                    sem.release()

        if not self.api_key:
            if self.fallback_local:
                if sem:
                    sem.release()
                return {"text": self._local_answer(prompt), "raw": None, "tool_calls": []}
            logger.error("OpenRouter not configured. Set OPENROUTER_API_KEY in env.")
            if sem:
                sem.release()
            raise RuntimeError("OpenRouter not configured. Set OPENROUTER_API_KEY in env.")
        flush_offline_queue(self.api_key, self.model, self.site, self.title, self.http_fallback_url, async_mode=True)

        try:
            msg_payload = to_openrouter_messages(messages) if messages else [{"role": "user", "content": prompt}]
            if self.use_sdk and tools:
                logger.warning("SDK tool-calls not wired; falling back to httpx.")
                self.use_sdk = False

            if self.use_sdk:
                async with OpenRouter(api_key=self.api_key) as client:  # type: ignore[arg-type]
                    try:
                        with start_span("llm.generate.sdk_async"):
                            resp = await client.chat.send_async(
                                model=self.model or OPENROUTER_MODEL,
                                messages=msg_payload,
                                max_tokens=max_tokens,
                            )
                        return {"text": self._extract_text(resp), "raw": resp, "tool_calls": []}
                    except Exception as e:
                        logger.warning("LLM SDK call failed (async): %s", e, exc_info=True)
                        if self.fallback_local:
                            return {"text": self._local_answer(prompt), "raw": None, "tool_calls": []}
                        if OFFLINE_QUEUE_ENABLED:
                            enqueue_offline_request(prompt, max_tokens)
                            return {"text": "[queued_offline]", "raw": None, "tool_calls": []}
                        raise

            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            if self.site:
                headers["HTTP-Referer"] = self.site
            if self.title:
                headers["X-Title"] = self.title
            payload = {
                "model": self.model or OPENROUTER_MODEL,
                "messages": msg_payload,
                "max_tokens": max_tokens,
                "stream": False,
            }
            if tools:
                payload["tools"] = tools
                if tool_choice:
                    payload["tool_choice"] = tool_choice
            async with httpx.AsyncClient(timeout=60.0) as client:
                try:
                    with start_span("llm.generate.httpx_async"):
                        resp = await client.post(self.http_fallback_url, json=payload, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                    logger.debug("LLM response (httpx): %s", data)
                    tool_calls = _extract_tool_calls(data)
                    return {"text": self._extract_text(data), "raw": data, "tool_calls": tool_calls}
                except httpx.HTTPStatusError as e:
                    delay = _retry_after_seconds(e.response)
                    if delay > 0:
                        logger.info("Rate limit/backoff (async) waiting %ss per headers", delay)
                        await asyncio.sleep(delay)
                    logger.warning("LLM httpx call failed (async): %s", e, exc_info=True)
                    if self.fallback_local:
                        return {"text": self._local_answer(prompt), "raw": None, "tool_calls": []}
                    if OFFLINE_QUEUE_ENABLED:
                        enqueue_offline_request(prompt, max_tokens)
                        return {"text": "[queued_offline]", "raw": None, "tool_calls": []}
                    raise
                except Exception as e:
                    logger.warning("LLM httpx call failed (async): %s", e, exc_info=True)
                    if self.fallback_local:
                        return {"text": self._local_answer(prompt), "raw": None, "tool_calls": []}
                    if OFFLINE_QUEUE_ENABLED:
                        enqueue_offline_request(prompt, max_tokens)
                        return {"text": "[queued_offline]", "raw": None, "tool_calls": []}
                    raise
        finally:
            if sem:
                sem.release()

    def generate_sync(self, prompt: str = "", *, messages: Optional[List[Union[LLMMessage, Dict[str, Any]]]] = None, max_tokens: int = 512, tools: Optional[list] = None, tool_choice: Optional[str] = None) -> str:
        """Synchronous generate using SDK when available, otherwise httpx. Returns plain text (for compatibility)."""
        sem = _sync_semaphore
        if sem:
            sem.acquire()
        try:
            if self.fallback_local:
                return self._local_answer(prompt)
            # 1) Prefer LM Studio local server when provider=local
            if self.provider == "local":
                if not self.lmstudio_url:
                    raise RuntimeError("PROVIDER=local mais LMSTUDIO_URL manquant.")
                return self._generate_lmstudio_sync(messages or [{"role": "user", "content": prompt}], max_tokens, tools, tool_choice)
            if not self.api_key:
                logger.error("OpenRouter not configured (sync).")
                raise RuntimeError("OpenRouter not configured. Set OPENROUTER_API_KEY in env.")
            flush_offline_queue(self.api_key, self.model, self.site, self.title, self.http_fallback_url, async_mode=False)

            msg_payload = to_openrouter_messages(messages) if messages else [{"role": "user", "content": prompt}]
            if self.use_sdk:
                with OpenRouter(api_key=self.api_key) as client:  # type: ignore[arg-type]
                    try:
                        with start_span("llm.generate.sdk_sync"):
                            resp = client.chat.send(  # type: ignore[attr-defined]
                                model=self.model or OPENROUTER_MODEL,
                                messages=msg_payload,
                                max_tokens=max_tokens,
                            )
                        return self._extract_text(resp)
                    except Exception as e:
                        logger.warning("LLM SDK call failed (sync): %s", e, exc_info=True)
                        if self.fallback_local:
                            return self._local_answer(prompt)
                        if OFFLINE_QUEUE_ENABLED:
                            enqueue_offline_request(prompt, max_tokens)
                            return "[queued_offline]"
                        raise

            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            if self.site:
                headers["HTTP-Referer"] = self.site
            if self.title:
                headers["X-Title"] = self.title
            payload = {
                "model": self.model or OPENROUTER_MODEL,
                "messages": msg_payload,
                "max_tokens": max_tokens,
                "stream": False,
            }
            if tools:
                payload["tools"] = tools
                if tool_choice:
                    payload["tool_choice"] = tool_choice
            try:
                with start_span("llm.generate.httpx_sync"):
                    resp = httpx.post(self.http_fallback_url, json=payload, headers=headers, timeout=60.0)
                resp.raise_for_status()
                data = resp.json()
                logger.debug("LLM sync response: %s", data)
                return self._extract_text(data)
            except httpx.HTTPStatusError as e:
                delay = _retry_after_seconds(e.response)
                if delay > 0:
                    logger.info("Rate limit/backoff (sync) waiting %ss per headers", delay)
                    time.sleep(delay)
                logger.warning("LLM httpx call failed (sync): %s", e, exc_info=True)
                if self.fallback_local:
                    return self._local_answer(prompt)
                if OFFLINE_QUEUE_ENABLED:
                    enqueue_offline_request(prompt, max_tokens)
                    return "[queued_offline]"
                raise
            except Exception as e:
                logger.warning("LLM httpx call failed (sync): %s", e, exc_info=True)
                if self.fallback_local:
                    return self._local_answer(prompt)
                if OFFLINE_QUEUE_ENABLED:
                    enqueue_offline_request(prompt, max_tokens)
                    return "[queued_offline]"
                raise
        finally:
            if sem:
                sem.release()

    def check_health_sync(self) -> dict:
        """Lightweight healthcheck against OpenRouter models endpoint (no tokens consumed)."""
        url = "https://openrouter.ai/api/v1/models"
        if not self.api_key:
            return {"ok": False, "error": "no_api_key"}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            resp = httpx.get(url, headers=headers, timeout=10.0)
            resp.raise_for_status()
            return {"ok": True, "status_code": resp.status_code}
        except Exception as e:
            logger.warning("LLM healthcheck failed: %s", e)
            return {"ok": False, "error": str(e)}

    @staticmethod
    def _extract_text(data: dict) -> str:
        """Extract text from OpenAI/OpenRouter-like responses or SDK objects."""
        # SDK response: pydantic model with choices list
        try:
            if hasattr(data, "choices"):
                choices = getattr(data, "choices")
                if choices:
                    ch0 = choices[0]
                    if hasattr(ch0, "message") and ch0.message and hasattr(ch0.message, "content"):
                        return ch0.message.content  # type: ignore[return-value]
                    if hasattr(ch0, "delta") and ch0.delta and hasattr(ch0.delta, "content"):
                        return ch0.delta.content  # type: ignore[return-value]
        except Exception:
            pass

        if isinstance(data, dict):
            if 'output' in data:
                return data['output']
            if 'text' in data:
                return data['text']
            if 'choices' in data and len(data['choices']) > 0:
                choice0 = data['choices'][0]
                if isinstance(choice0, dict):
                    if 'text' in choice0:
                        return choice0['text']
                    if 'message' in choice0 and isinstance(choice0['message'], dict) and 'content' in choice0['message']:
                        return choice0['message']['content']
                    if 'delta' in choice0 and isinstance(choice0['delta'], dict) and 'content' in choice0['delta']:
                        return choice0['delta']['content']
        return str(data)

    def _local_answer(self, prompt: str) -> str:
        """Very small local fallback when remote LLM is unavailable."""
        return f"[local-fallback] {prompt[:200]}"

    async def _generate_lmstudio_async(self, messages: List[Union[LLMMessage, Dict[str, Any]]], max_tokens: int, tools: Optional[list], tool_choice: Optional[str]) -> dict:
        """Call a local LM Studio server (OpenAI-compatible) asynchronously."""
        msg_payload = to_openrouter_messages(messages)
        payload = {
            "model": self.lmstudio_model or self.model or OPENROUTER_MODEL,
            "messages": msg_payload,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(self.lmstudio_url, json=payload, headers={"Content-Type": "application/json"})
            resp.raise_for_status()
            data = resp.json()
            return {"text": self._extract_text(data), "raw": data, "tool_calls": _extract_tool_calls(data)}

    def _generate_lmstudio_sync(self, messages: List[Union[LLMMessage, Dict[str, Any]]], max_tokens: int, tools: Optional[list], tool_choice: Optional[str]) -> str:
        """Call a local LM Studio server (OpenAI-compatible) synchronously."""
        msg_payload = to_openrouter_messages(messages)
        payload = {
            "model": self.lmstudio_model or self.model or OPENROUTER_MODEL,
            "messages": msg_payload,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice
        resp = httpx.post(self.lmstudio_url, json=payload, headers={"Content-Type": "application/json"}, timeout=60.0)
        resp.raise_for_status()
        data = resp.json()
        return self._extract_text(data)


def enqueue_offline_request(prompt: str, max_tokens: int):
    """Persist a failed request locally for later retry."""
    try:
        OFFLINE_QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OFFLINE_QUEUE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"prompt": prompt, "max_tokens": max_tokens}) + "\n")
        logger.info("Queued LLM request offline (file=%s)", OFFLINE_QUEUE_FILE)
    except Exception as e:  # pragma: no cover - best effort
        logger.warning("Failed to queue offline request: %s", e)


def flush_offline_queue(api_key: str, model: str, site: str, title: str, url: str, async_mode: bool = False):
    """Try to replay offline queued requests when connectivity is back."""
    if not OFFLINE_QUEUE_ENABLED or not OFFLINE_QUEUE_FILE.exists():
        return
    try:
        lines = OFFLINE_QUEUE_FILE.read_text(encoding="utf-8").splitlines()
        OFFLINE_QUEUE_FILE.unlink()
    except Exception:
        return
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if site:
        headers["HTTP-Referer"] = site
    if title:
        headers["X-Title"] = title
    if async_mode:
        import asyncio
        async def _send():
            async with httpx.AsyncClient(timeout=30.0) as client:
                for line in lines:
                    try:
                        payload = json.loads(line)
                        payload_http = {
                            "model": model or OPENROUTER_MODEL,
                            "messages": [{"role": "user", "content": payload.get("prompt")}],
                            "max_tokens": payload.get("max_tokens", 256),
                            "stream": False,
                        }
                        await client.post(url, json=payload_http, headers=headers)
                    except Exception:
                        enqueue_offline_request(payload.get("prompt", ""), payload.get("max_tokens", 256))
        try:
            asyncio.create_task(_send())
        except Exception:
            pass
    else:
        for line in lines:
            try:
                payload = json.loads(line)
                payload_http = {
                    "model": model or OPENROUTER_MODEL,
                    "messages": [{"role": "user", "content": payload.get("prompt")}],
                    "max_tokens": payload.get("max_tokens", 256),
                    "stream": False,
                }
                httpx.post(url, json=payload_http, headers=headers, timeout=30.0)
            except Exception:
                enqueue_offline_request(payload.get("prompt", ""), payload.get("max_tokens", 256))


def _retry_after_seconds(response: httpx.Response) -> float:
    """Parse Retry-After header if present."""
    if not response:
        return 0.0
    val = response.headers.get("Retry-After") or response.headers.get("retry-after")
    if not val:
        return 0.0
    try:
        return float(val)
    except Exception:
        return 0.0


    def _local_answer(self, prompt: str) -> str:
        """Very small local fallback when remote LLM is unavailable."""
        # naive echo-style answer
        return f"[local-fallback] {prompt[:200]}"


def _extract_tool_calls(data: dict) -> list:
    """Extract tool_calls from OpenAI/OpenRouter-style responses."""
    try:
        if isinstance(data, dict):
            choices = data.get("choices") or []
            tool_calls = []
            for ch in choices:
                msg = ch.get("message") if isinstance(ch, dict) else None
                if msg and msg.get("tool_calls"):
                    tool_calls.extend(msg.get("tool_calls"))
            return tool_calls
    except Exception:
        return []
    return []
