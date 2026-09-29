"""OpenRouter client for the red-teaming defense eval project.

Uses httpx against OpenRouter's OpenAI-compatible chat completions endpoint.
Mirrors the semantics of the suite's LLMClient: temperature 0, 30s timeout,
and EvaluationError on any failure.

Auth: the user's OpenRouter key lives in the secure vault as
``custom.openrouter``. The key (or a surrogate for it) is never printed or
persisted; it is sent only as a Bearer header to openrouter.ai. Passing
``api_key`` explicitly bypasses the vault, which keeps unit tests hermetic.
"""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from typing import Any

import httpx

from adversarial_prompt_suite.exceptions import EvaluationError

# Hosts that may receive the auth surrogate. Nothing else gets it.
_OPENROUTER_HOSTS = ("openrouter.ai",)

# Lazily added so importing this module never touches the vault.
_VAULT_BIN_DIR = "/opt/hatch/skills/skill-creator/bin"


class OpenRouterClient:
    """Thin async wrapper around OpenRouter's /chat/completions endpoint."""

    def __init__(
        self,
        base_url: str = "https://openrouter.ai/api/v1",
        api_key: str | None = None,
        model_id: str = "qwen/qwen3.8-27b:free",
        system_prompt: str | None = None,
        timeout_s: float = 30.0,
        max_tokens: int = 512,
    ) -> None:
        """Initialise the OpenRouter client.

        Args:
            base_url: OpenAI-compatible API base URL.
            api_key: Explicit bearer token. If None, the key is fetched from
                the secure vault (custom.openrouter) at request time.
            model_id: Model identifier to pass in requests.
            system_prompt: Optional system message prepended before the user
                message. Used directly as the D1 system-prompt-hardening
                defense; no separate wrapper class is needed.
            timeout_s: Per-request timeout in seconds.
            max_tokens: Max completion tokens per request.
        """
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model_id = model_id
        self._system_prompt = system_prompt
        self._timeout = httpx.Timeout(timeout_s)
        self._max_tokens = max_tokens
        # Cache for the vault surrogate. _auth_token() does a synchronous
        # unix-socket round-trip to authd with no timeout; calling it on
        # every request risks hanging the process if authd stalls (observed
        # as a reproducible hang in long runs). Fetch once, reuse in memory
        # for the life of the client. Never written to disk or logs.
        self._cached_token: str | None = api_key

    @property
    def model_id(self) -> str:
        """Return the model identifier used by this client."""
        return self._model_id

    def _auth_token(self) -> str:
        """Return the bearer token, from the explicit key or the vault.

        The vault surrogate is fetched once and cached in memory for the
        life of the client, because the underlying authd round-trip is
        synchronous with no timeout.

        Raises:
            EvaluationError: If the vault lookup fails or the credential
                placement is unexpected.
        """
        if self._cached_token is not None:
            return self._cached_token

        if _VAULT_BIN_DIR not in sys.path:
            sys.path.insert(0, _VAULT_BIN_DIR)
        from dynamic_credentials import (  # noqa: E402
            DynamicCredentialError,
            dynamic_credential_entry,
            ensure_allowed_url,
        )

        url = f"{self._base_url}/chat/completions"
        ensure_allowed_url(url, list(_OPENROUTER_HOSTS))
        try:
            entry = dynamic_credential_entry("custom.openrouter", "access_token")
        except DynamicCredentialError as exc:
            raise EvaluationError(f"OpenRouter auth failed: {exc}") from exc

        if entry.get("placement") != "bearer_header":
            raise EvaluationError(
                "OpenRouter auth failed: unexpected credential placement "
                f"{entry.get('placement')!r}"
            )
        surrogate = str(entry["surrogate"]).strip()
        if not surrogate.startswith("hsurr:"):
            raise EvaluationError("OpenRouter auth failed: vault value is not a surrogate")
        self._cached_token = surrogate
        return surrogate

    async def complete(self, prompt: str, response_format: dict | None = None) -> str:
        """Send a single chat completion request.

        Args:
            prompt: The prompt text to send as the user message.
            response_format: Optional OpenAI-style response format (e.g.
                ``{"type": "json_object"}``). Passed through to the API;
                callers that need structured output (the LLM judge) use it.

        Returns:
            The model's response text.

        Raises:
            EvaluationError: On network failure, non-2xx response, timeout,
                auth failure, or an unexpected response shape.
        """
        url = f"{self._base_url}/chat/completions"
        messages = [{"role": "user", "content": prompt}]
        if self._system_prompt:
            messages.insert(0, {"role": "system", "content": self._system_prompt})

        payload: dict[str, Any] = {
            "model": self._model_id,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self._max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.post(
                    url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self._auth_token()}",
                        "Content-Type": "application/json",
                    },
                )
                response.raise_for_status()
            except httpx.TimeoutException as exc:
                raise EvaluationError(f"Request timed out: {exc}") from exc
            except httpx.HTTPStatusError as exc:
                raise EvaluationError(
                    f"API returned {exc.response.status_code}: {exc.response.text[:200]}"
                ) from exc
            except httpx.RequestError as exc:
                raise EvaluationError(f"Network error: {exc}") from exc

        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise EvaluationError(f"Unexpected API response structure: {data}") from exc
        if not isinstance(content, str):
            raise EvaluationError(f"Unexpected API response structure: {data}")
        return content


class _CompletionsNamespace:
    """Shim for the OpenAI SDK's client.chat.completions object."""

    def __init__(self, adapter: JudgeAdapter) -> None:
        self._adapter = adapter

    def create(self, **kwargs: Any) -> SimpleNamespace:
        return self._adapter._create_completion(**kwargs)


class _ChatNamespace:
    """Shim for the OpenAI SDK's client.chat object."""

    def __init__(self, adapter: JudgeAdapter) -> None:
        self.completions = _CompletionsNamespace(adapter)


class JudgeAdapter:
    """OpenAI-SDK-shaped adapter backed by an async completion client.

    The suite's LLMJudgeClassifier calls
    ``client.chat.completions.create(model=..., messages=..., temperature=0,
    response_format=...)`` and always passes ``model="gpt-4o-mini"``. This
    adapter forwards the prompt text to the wrapped client and ignores the
    passed model name; the wrapped client (constructed with the judge model
    id by the runner) decides which model answers. It adapts, not forks:
    LLMJudgeClassifier itself is untouched.
    """

    def __init__(self, client: Any, judge_model_id: str) -> None:
        """Initialise the adapter.

        Args:
            client: Any object with async ``complete(prompt) -> str``.
            judge_model_id: The judge model id the wrapped client uses.
                Exposed via ``.model_id`` so results record the real model.
        """
        self._client = client
        self._judge_model_id = judge_model_id
        self.chat = _ChatNamespace(self)

    @property
    def model_id(self) -> str:
        """Return the judge model identifier used by this adapter."""
        return self._judge_model_id

    def _create_completion(
        self,
        model: str | None = None,
        messages: list | None = None,
        temperature: int = 0,
        response_format: dict | None = None,
    ) -> SimpleNamespace:
        """Serve client.chat.completions.create() synchronously.

        The call is synchronous because LLMJudgeClassifier's call sites are
        synchronous. The wrapped client's coroutine is run to completion here.

        Args:
            model: Ignored; the wrapped client carries the judge model id.
            messages: OpenAI-style message list; contents are concatenated.
            temperature: Ignored; the wrapped client always uses temperature 0.
            response_format: Ignored; JSON output is requested in the prompt.

        Returns:
            Namespace with ``.choices[0].message.content``.
        """
        parts = []
        for message in messages or []:
            content = message.get("content", "")
            parts.append(str(content))
        prompt = "\n".join(parts)
        # response_format is forwarded so the judge gets clean JSON back.
        # asyncio.run is safe here because the suite calls this adapter from
        # synchronous code; the runner invokes classify_batch in a worker
        # thread (asyncio.to_thread) precisely so no event loop is running.
        # The TypeError fallback keeps the adapter compatible with clients
        # whose complete() takes only a prompt (e.g. test fakes).
        try:
            text = asyncio.run(self._client.complete(prompt, response_format=response_format))
        except TypeError:
            text = asyncio.run(self._client.complete(prompt))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])
