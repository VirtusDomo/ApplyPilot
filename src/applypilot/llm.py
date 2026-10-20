"""
Unified LLM client for ApplyPilot.

Auto-detects provider from environment (see ``config.llm_provider``):
  LLM_PROVIDER    -> force one of: claude, gemini, openai, local
  LLM_URL         -> Local llama.cpp / Ollama compatible endpoint
  GEMINI_API_KEY  -> Google Gemini (default: gemini-2.0-flash)
  OPENAI_API_KEY  -> OpenAI (default: gpt-4o-mini)
  (none of these) -> Claude Code CLI (``claude -p``) on your Claude
                     subscription, if ``claude`` is on PATH

LLM_MODEL env var overrides the model name for the API providers. With
Claude, each task gets its own model (Haiku for scoring and extraction,
Sonnet for writing and fabrication checks); override all of them with
CLAUDE_MODEL or one task with CLAUDE_MODEL_<TASK>, e.g. CLAUDE_MODEL_TAILOR=opus.
"""

import json
import logging
import os
import subprocess
import tempfile
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from applypilot import config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

def _detect_provider() -> tuple[str, str, str]:
    """Return (base_url, model, api_key) for an HTTP provider.

    Reads env at call time (not module import time) so that load_env() called
    in _bootstrap() is always visible here.
    """
    provider = config.llm_provider()
    model_override = os.environ.get("LLM_MODEL", "")

    if provider == "gemini":
        return (
            "https://generativelanguage.googleapis.com/v1beta/openai",
            model_override or "gemini-2.0-flash",
            os.environ.get("GEMINI_API_KEY", ""),
        )

    if provider == "openai":
        return (
            "https://api.openai.com/v1",
            model_override or "gpt-4o-mini",
            os.environ.get("OPENAI_API_KEY", ""),
        )

    if provider == "local":
        return (
            os.environ.get("LLM_URL", "").rstrip("/"),
            model_override or "local-model",
            os.environ.get("LLM_API_KEY", ""),
        )

    raise RuntimeError(
        "No LLM provider configured. Install Claude Code (claude.ai/code) "
        "or set GEMINI_API_KEY, OPENAI_API_KEY, or LLM_URL in your environment."
    )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

_MAX_RETRIES = 5
_TIMEOUT = 120  # seconds

# Base wait on first 429/503 (doubles each retry, caps at 60s).
# Gemini free tier is 15 RPM = 4s minimum between requests; 10s gives headroom.
_RATE_LIMIT_BASE_WAIT = 10


_GEMINI_COMPAT_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
_GEMINI_NATIVE_BASE = "https://generativelanguage.googleapis.com/v1beta"


class LLMClient:
    """Thin LLM client supporting OpenAI-compatible and native Gemini endpoints.

    For Gemini keys, starts on the OpenAI-compat layer. On a 403 (which
    happens with preview/experimental models not exposed via compat), it
    automatically switches to the native generateContent API and stays there
    for the lifetime of the process.
    """

    def __init__(self, base_url: str, model: str, api_key: str) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self._client = httpx.Client(timeout=_TIMEOUT)
        # True once we've confirmed the native Gemini API works for this model
        self._use_native_gemini: bool = False
        self._is_gemini: bool = base_url.startswith(_GEMINI_COMPAT_BASE)

    # -- Native Gemini API --------------------------------------------------

    def _chat_native_gemini(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call the native Gemini generateContent API.

        Used automatically when the OpenAI-compat endpoint returns 403,
        which happens for preview/experimental models not exposed via compat.

        Converts OpenAI-style messages to Gemini's contents/systemInstruction
        format transparently.
        """
        contents: list[dict] = []
        system_parts: list[dict] = []

        for msg in messages:
            role = msg["role"]
            text = msg.get("content", "")
            if role == "system":
                system_parts.append({"text": text})
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": text}]})
            elif role == "assistant":
                # Gemini uses "model" instead of "assistant"
                contents.append({"role": "model", "parts": [{"text": text}]})

        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}

        url = f"{_GEMINI_NATIVE_BASE}/models/{self.model}:generateContent"
        resp = self._client.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            params={"key": self.api_key},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    # -- OpenAI-compat API --------------------------------------------------

    def _chat_compat(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call the OpenAI-compatible endpoint."""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        resp = self._client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
        )

        # 403 on Gemini compat = model not available on compat layer.
        # Raise a specific sentinel so chat() can switch to native API.
        if resp.status_code == 403 and self._is_gemini:
            raise _GeminiCompatForbidden(resp)

        return self._handle_compat_response(resp)

    @staticmethod
    def _handle_compat_response(resp: httpx.Response) -> str:
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    # -- public API ---------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """Send a chat completion request and return the assistant message text."""
        # Qwen3 optimization: prepend /no_think to skip chain-of-thought
        # reasoning, saving tokens on structured extraction tasks.
        if "qwen" in self.model.lower() and messages:
            first = messages[0]
            if first.get("role") == "user" and not first["content"].startswith("/no_think"):
                messages = [{"role": first["role"], "content": f"/no_think\n{first['content']}"}] + messages[1:]

        for attempt in range(_MAX_RETRIES):
            try:
                # Route to native Gemini if we've already confirmed it's needed
                if self._use_native_gemini:
                    return self._chat_native_gemini(messages, temperature, max_tokens)

                return self._chat_compat(messages, temperature, max_tokens)

            except _GeminiCompatForbidden as exc:
                # Model not available on OpenAI-compat layer — switch to native.
                log.warning(
                    "Gemini compat endpoint returned 403 for model '%s'. "
                    "Switching to native generateContent API. "
                    "(Preview/experimental models are often compat-only on native.)",
                    self.model,
                )
                self._use_native_gemini = True
                # Retry immediately with native — don't count as a rate-limit wait
                try:
                    return self._chat_native_gemini(messages, temperature, max_tokens)
                except httpx.HTTPStatusError as native_exc:
                    raise RuntimeError(
                        f"Both Gemini endpoints failed. Compat: 403 Forbidden. "
                        f"Native: {native_exc.response.status_code} — "
                        f"{native_exc.response.text[:200]}"
                    ) from native_exc

            except httpx.HTTPStatusError as exc:
                resp = exc.response
                if resp.status_code in (429, 503) and attempt < _MAX_RETRIES - 1:
                    # Respect Retry-After header if provided (Gemini sends this).
                    retry_after = (
                        resp.headers.get("Retry-After")
                        or resp.headers.get("X-RateLimit-Reset-Requests")
                    )
                    if retry_after:
                        try:
                            wait = float(retry_after)
                        except (ValueError, TypeError):
                            wait = _RATE_LIMIT_BASE_WAIT * (2 ** attempt)
                    else:
                        wait = min(_RATE_LIMIT_BASE_WAIT * (2 ** attempt), 60)

                    log.warning(
                        "LLM rate limited (HTTP %s). Waiting %ds before retry %d/%d. "
                        "Tip: Gemini free tier = 15 RPM. Consider a paid account "
                        "or switching to a local model.",
                        resp.status_code, wait, attempt + 1, _MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue
                raise

            except httpx.TimeoutException:
                if attempt < _MAX_RETRIES - 1:
                    wait = min(_RATE_LIMIT_BASE_WAIT * (2 ** attempt), 60)
                    log.warning(
                        "LLM request timed out, retrying in %ds (attempt %d/%d)",
                        wait, attempt + 1, _MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue
                raise

        raise RuntimeError("LLM request failed after all retries")

    def ask(self, prompt: str, **kwargs) -> str:
        """Convenience: single user prompt -> assistant response."""
        return self.chat([{"role": "user", "content": prompt}], **kwargs)

    def close(self) -> None:
        self._client.close()


# ---------------------------------------------------------------------------
# Claude Code CLI
# ---------------------------------------------------------------------------

# Default model per task. Scoring and extraction run on every job, so they use
# the cheapest model; writing and the fabrication judge are worth Sonnet.
CLAUDE_TASK_MODELS: dict[str, str] = {
    "default": "haiku",
    "extract": "haiku",
    "score": "haiku",
    "tailor": "sonnet",
    "judge": "sonnet",
    "cover": "sonnet",
}

# Reasoning effort per task. Scoring and extraction are structured, run on
# every job, and measured about 2x faster at "low" with the same answers.
# Tasks not listed use the CLI default.
CLAUDE_TASK_EFFORT: dict[str, str] = {
    "extract": "low",
    "score": "low",
}

_CLAUDE_TIMEOUT = 300  # seconds per call
_CLAUDE_MAX_PARALLEL = 2  # be gentle with subscription rate limits
_claude_slots = threading.BoundedSemaphore(_CLAUDE_MAX_PARALLEL)


# Phrases the CLI uses when the subscription's usage window is exhausted.
_USAGE_LIMIT_MARKERS = ("usage limit", "limit reached", "hit your limit", "limit will reset",
                        "resets at", "out of extra usage")


class ClaudeUsageLimitError(RuntimeError):
    """The Claude subscription's usage limit was hit. Retrying won't help."""


def claude_model_for(task: str) -> str:
    """Resolve the model alias for a task.

    CLAUDE_MODEL_<TASK> beats CLAUDE_MODEL, which beats the defaults. LLM_MODEL
    is ignored here: it usually names a Gemini/OpenAI model from an older .env.
    """
    return (
        os.environ.get(f"CLAUDE_MODEL_{task.upper()}")
        or os.environ.get("CLAUDE_MODEL")
        or CLAUDE_TASK_MODELS.get(task, CLAUDE_TASK_MODELS["default"])
    )


def claude_effort_for(task: str) -> str | None:
    """Resolve the reasoning effort for a task (CLAUDE_EFFORT_<TASK> overrides the default)."""
    return os.environ.get(f"CLAUDE_EFFORT_{task.upper()}") or CLAUDE_TASK_EFFORT.get(task)


def build_claude_cmd(claude_path: str, model: str, system_prompt_file: str | None,
                     effort: str | None = None) -> list[str]:
    """Plain text-in/text-out Claude call: no tools, MCP, hooks, CLAUDE.md or skills.

    ``--safe-mode`` drops every user customization (your hooks would otherwise
    fire, and your global CLAUDE.md would leak into scoring prompts) while
    keeping subscription auth. ``--tools ""`` removes all built-in tools, so a
    job description can't make the model touch your machine.
    """
    cmd = [
        claude_path, "-p",
        "--model", model,
        "--safe-mode",
        "--tools", "",
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--no-session-persistence",
        "--output-format", "json",
    ]
    if system_prompt_file:
        cmd += ["--system-prompt-file", system_prompt_file]
    if effort:
        cmd += ["--effort", effort]
    return cmd


def _render_messages(messages: list[dict]) -> tuple[str, str]:
    """Split OpenAI-style messages into (system_prompt, prompt_text)."""
    system = "\n\n".join(m.get("content") or "" for m in messages if m["role"] == "system")
    turns = [m for m in messages if m["role"] != "system"]
    if len(turns) == 1:
        return system, turns[0].get("content") or ""
    # Multi-turn history has no CLI equivalent, so replay it as a transcript.
    return system, "\n\n".join(f"[{m['role'].upper()}]\n{m.get('content') or ''}" for m in turns)


_usage_lock = threading.Lock()


def usage_log_path() -> Path:
    return config.LOG_DIR / "llm_usage.jsonl"


def record_usage(kind: str, result: dict) -> None:
    """Append one Claude call's usage (from a CLI ``result`` message) to the usage log.

    ``cost_usd`` is the API-list-price equivalent the CLI reports. On a
    subscription nothing is billed per call, but it is the best available
    measure of how fast a run eats into the plan's usage limits.
    """
    usage = result.get("usage") or {}
    entry = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "kind": kind,
        "models": sorted((result.get("modelUsage") or {}).keys()),
        "cost_usd": result.get("total_cost_usd") or 0,
        "input_tokens": usage.get("input_tokens") or 0,
        "output_tokens": usage.get("output_tokens") or 0,
        "cache_read": usage.get("cache_read_input_tokens") or 0,
        "cache_create": usage.get("cache_creation_input_tokens") or 0,
        "turns": result.get("num_turns") or 0,
        "ok": not result.get("is_error"),
    }
    try:
        path = usage_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with _usage_lock, path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except OSError:
        log.debug("Could not write usage log", exc_info=True)


def usage_summary(days: int = 7) -> dict[str, dict]:
    """Totals per call kind over the last ``days`` days: {kind: {calls, cost_usd, tokens}}."""
    cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    totals: dict[str, dict] = {}
    path = usage_log_path()
    if not path.exists():
        return totals
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("ts", "") < cutoff:
            continue
        t = totals.setdefault(entry.get("kind", "?"), {"calls": 0, "cost_usd": 0.0, "tokens": 0})
        t["calls"] += 1
        t["cost_usd"] += entry.get("cost_usd") or 0
        t["tokens"] += sum(entry.get(k) or 0 for k in
                           ("input_tokens", "output_tokens", "cache_read", "cache_create"))
    return totals


def parse_claude_output(stdout: str, kind: str = "llm") -> str:
    """Return the reply text from ``--output-format json`` output, or raise.

    Every parsed result, success or error, is recorded in the usage log.
    """
    try:
        data = json.loads(stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise RuntimeError(f"Unreadable Claude CLI output: {stdout[:300]!r}") from exc
    record_usage(kind, data)

    if not data.get("is_error") and data.get("subtype") == "success":
        return data.get("result", "")

    detail = str(data.get("result") or data.get("subtype") or "unknown error")
    if any(marker in detail.lower() for marker in _USAGE_LIMIT_MARKERS):
        raise ClaudeUsageLimitError(f"Claude usage limit reached: {detail[:300]}")
    raise RuntimeError(f"Claude CLI error ({data.get('api_error_status')}): {detail[:300]}")


class ClaudeCLIClient:
    """LLM client that runs prompts through the local Claude Code CLI.

    Uses the user's Claude subscription (Pro/Max) instead of an API key.
    ``temperature`` and ``max_tokens`` are accepted for interface parity but
    the CLI does not expose them.
    """

    def __init__(self, claude_path: str, model: str, task: str = "default") -> None:
        self.claude_path = claude_path
        self.model = model
        self.task = task  # label for the usage log

    def _run_once(self, system: str, prompt: str) -> str:
        sp_file = None
        if system:
            fd, sp_file = tempfile.mkstemp(prefix="applypilot-sys-", suffix=".md")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(system)
        try:
            with _claude_slots:
                proc = subprocess.run(
                    build_claude_cmd(self.claude_path, self.model, sp_file, claude_effort_for(self.task)),
                    input=prompt,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=config.claude_subprocess_env(),
                    cwd=tempfile.gettempdir(),  # keep project settings out
                    timeout=_CLAUDE_TIMEOUT,
                    check=False,  # errors arrive as JSON on stdout
                )
        finally:
            if sp_file:
                Path(sp_file).unlink(missing_ok=True)

        if not proc.stdout.strip():
            raise RuntimeError(
                f"Claude CLI exited {proc.returncode} with no output: {proc.stderr[:300]}"
            )
        return parse_claude_output(proc.stdout, kind=self.task)

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """Send a chat request and return the assistant message text."""
        system, prompt = _render_messages(messages)
        timeouts = 0
        for attempt in range(_MAX_RETRIES):
            try:
                return self._run_once(system, prompt)
            except ClaudeUsageLimitError:
                raise
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                timeouts += isinstance(exc, subprocess.TimeoutExpired)
                if attempt == _MAX_RETRIES - 1 or timeouts > 1:
                    raise
                wait = min(_RATE_LIMIT_BASE_WAIT * (2 ** attempt), 60)
                log.warning("Claude call failed (%s); retrying in %ds (%d/%d)",
                            exc, wait, attempt + 1, _MAX_RETRIES)
                time.sleep(wait)
        raise RuntimeError("Claude request failed after all retries")

    def ask(self, prompt: str, **kwargs) -> str:
        """Convenience: single user prompt -> assistant response."""
        return self.chat([{"role": "user", "content": prompt}], **kwargs)

    def close(self) -> None:
        pass


class _GeminiCompatForbidden(Exception):
    """Sentinel: Gemini OpenAI-compat returned 403. Switch to native API."""
    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        super().__init__(f"Gemini compat 403: {response.text[:200]}")


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: LLMClient | None = None
_claude_clients: dict[str, ClaudeCLIClient] = {}
_lock = threading.Lock()


def get_client(task: str = "default") -> LLMClient | ClaudeCLIClient:
    """Return the client for a task.

    ``task`` is a key of ``CLAUDE_TASK_MODELS`` and only matters for the Claude
    provider, which picks a model per task. HTTP providers share one singleton.
    """
    global _instance
    with _lock:
        if config.llm_provider() == "claude":
            model = claude_model_for(task)
            client = _claude_clients.get(task)
            if client is None or client.model != model:
                log.info("LLM provider: Claude Code CLI  task: %s  model: %s", task, model)
                client = ClaudeCLIClient(config.get_claude_path() or "claude", model, task)
                _claude_clients[task] = client
            return client

        if _instance is None:
            base_url, model, api_key = _detect_provider()
            log.info("LLM provider: %s  model: %s", base_url, model)
            _instance = LLMClient(base_url, model, api_key)
        return _instance
