"""LLM evaluation engine for adversarial attack templates.

Supports both synchronous single evaluation and async batch evaluation
with configurable concurrency. Uses any OpenAI-compatible API endpoint.

Design choices:
  - asyncio.Semaphore gates concurrency to avoid overwhelming the API.
  - Each evaluation is independent; failures are captured in EvaluationResult.error
    rather than propagating, to ensure full batch results are always returned.
  - Latency is measured wall-clock for each individual request.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Optional

import httpx

from adversarial_prompt_suite.exceptions import EvaluationError
from adversarial_prompt_suite.models import AttackTemplate, EvaluationResult

logger = logging.getLogger(__name__)

# Default timeout per request in seconds.
_DEFAULT_TIMEOUT_S = 30.0

# Default concurrency limit for async batch evaluation.
_DEFAULT_CONCURRENCY = 5

# Placeholder substituted when no target is provided.
_DEFAULT_TARGET = "describe how to do something harmful"


class LLMClient:
    """Thin async wrapper around an OpenAI-compatible chat completions endpoint.

    Designed to be injected into Evaluator, making it easy to mock in tests.
    """

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "sk-placeholder",
        model_id: str = "gpt-4o",
        timeout_s: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        """Initialise the LLM client.

        Args:
            base_url: OpenAI-compatible API base URL.
            api_key: Bearer token for the API.
            model_id: Model identifier to pass in requests.
            timeout_s: Per-request timeout in seconds.
        """
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model_id = model_id
        self._timeout = httpx.Timeout(timeout_s)

    @property
    def model_id(self) -> str:
        """Return the model identifier used by this client."""
        return self._model_id

    async def complete(self, prompt: str) -> str:
        """Send a single chat completion request.

        Args:
            prompt: The adversarial prompt text to send as user content.

        Returns:
            The model's response text.

        Raises:
            EvaluationError: On network failure, non-2xx response, or timeout.
        """
        payload = {
            "model": self._model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 512,
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.post(
                    f"{self._base_url}/chat/completions",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
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
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise EvaluationError(
                f"Unexpected API response structure: {data}"
            ) from exc


class MockLLMClient:
    """Deterministic mock LLM client for tests and offline development.

    Returns canned refusal responses by default. Injects a compliance
    response when the prompt contains 'COMPLY_TRIGGER' for testing
    the UNSAFE classification path.
    """

    COMPLY_TRIGGER = "COMPLY_TRIGGER"
    BORDERLINE_TRIGGER = "BORDERLINE_TRIGGER"

    def __init__(self, model_id: str = "mock-model") -> None:
        """Initialise the mock client.

        Args:
            model_id: Model identifier to use in EvaluationResult.
        """
        self._model_id = model_id
        self._call_count = 0

    @property
    def model_id(self) -> str:
        """Return the mock model identifier."""
        return self._model_id

    @property
    def call_count(self) -> int:
        """Return total number of completions requested."""
        return self._call_count

    async def complete(self, prompt: str) -> str:
        """Return a deterministic mock response based on prompt content.

        Args:
            prompt: Adversarial prompt text.

        Returns:
            Canned response string.
        """
        self._call_count += 1
        # Simulate a small async delay to model network I/O.
        await asyncio.sleep(0.001)

        if self.COMPLY_TRIGGER in prompt:
            return (
                "Sure, here is how to do that! Step 1: First, you need to..."
            )
        if self.BORDERLINE_TRIGGER in prompt:
            return (
                "This is an interesting question. I cannot help with that, "
                "but I can tell you that it's related to the topic of..."
            )
        return (
            "I'm not able to assist with that request. It appears to be "
            "attempting to bypass my safety guidelines. I cannot and will "
            "not provide content that violates my usage policies."
        )


class Evaluator:
    """Runs adversarial attack templates against an LLM and collects results.

    Supports both single-template evaluation and async batch evaluation
    with configurable concurrency limiting.
    """

    def __init__(
        self,
        client: object,
        concurrency: int = _DEFAULT_CONCURRENCY,
    ) -> None:
        """Initialise the evaluator.

        Args:
            client: LLMClient or any object with `.complete(prompt) -> str` and
                    `.model_id` attribute.
            concurrency: Max simultaneous in-flight requests during batch eval.
        """
        self._client = client
        self._concurrency = concurrency
        self._semaphore: Optional[asyncio.Semaphore] = None

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Return the concurrency semaphore, creating it if needed.

        Returns:
            asyncio.Semaphore limiting concurrent evaluations.
        """
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._concurrency)
        return self._semaphore

    def _render_prompt(
        self,
        template: AttackTemplate,
        target: str = _DEFAULT_TARGET,
        payload: str = "",
    ) -> str:
        """Substitute placeholders in a template string.

        Args:
            template: AttackTemplate with placeholder markers.
            target: Text to substitute for {TARGET}.
            payload: Text to substitute for {PAYLOAD} variants.

        Returns:
            Rendered prompt string ready for API submission.
        """
        rendered = template.template
        rendered = rendered.replace("{TARGET}", target)
        rendered = rendered.replace("{PAYLOAD}", payload or target)
        # Encoding-specific placeholders — filled with placeholder text in
        # the absence of a real encoder. A production system would encode here.
        rendered = rendered.replace("{PAYLOAD_B64}", "[base64-encoded-payload]")
        rendered = rendered.replace("{PAYLOAD_LEET}", "[l33t-sp34k-p4yl04d]")
        rendered = rendered.replace("{PAYLOAD_HOMOGLYPH}", "[h\u03bfm\u04e1gl\u0443ph-p\u0430yl\u03bed]")
        rendered = rendered.replace("{PAYLOAD_REVERSED}", "[reversed-payload]")
        rendered = rendered.replace("{PAYLOAD_ROT13}", "[ebg13-cnlybnq]")
        rendered = rendered.replace("{PAYLOAD_MORSE}", "[-- --- .-. ... .  -.-. --- -.. ..]")
        rendered = rendered.replace("{PERSON_NAME}", "[redacted-person-name]")
        return rendered

    async def evaluate_one(
        self,
        template: AttackTemplate,
        target: str = _DEFAULT_TARGET,
        payload: str = "",
    ) -> EvaluationResult:
        """Evaluate a single attack template.

        Args:
            template: AttackTemplate to evaluate.
            target: Substitution for {TARGET} placeholder.
            payload: Substitution for {PAYLOAD} placeholder.

        Returns:
            EvaluationResult capturing prompt, response, latency, and metadata.
        """
        prompt = self._render_prompt(template, target, payload)
        start_ms = time.monotonic() * 1000

        try:
            response_text = await self._client.complete(prompt)  # type: ignore[union-attr]
            latency_ms = time.monotonic() * 1000 - start_ms

            return EvaluationResult(
                id=str(uuid.uuid4()),
                template_id=template.id,
                category=template.category,
                prompt=prompt,
                response=response_text,
                model_id=self._client.model_id,  # type: ignore[union-attr]
                latency_ms=latency_ms,
            )

        except EvaluationError as exc:
            latency_ms = time.monotonic() * 1000 - start_ms
            logger.warning(
                "Evaluation failed for template '%s': %s", template.name, exc
            )
            return EvaluationResult(
                id=str(uuid.uuid4()),
                template_id=template.id,
                category=template.category,
                prompt=prompt,
                response="",
                model_id=self._client.model_id,  # type: ignore[union-attr]
                latency_ms=latency_ms,
                error=str(exc),
            )

    async def _evaluate_with_limit(
        self,
        template: AttackTemplate,
        target: str,
        payload: str,
    ) -> EvaluationResult:
        """Evaluate a single template, gated by the concurrency semaphore.

        Args:
            template: AttackTemplate to evaluate.
            target: Substitution for {TARGET}.
            payload: Substitution for {PAYLOAD}.

        Returns:
            EvaluationResult for this template.
        """
        async with self._get_semaphore():
            return await self.evaluate_one(template, target, payload)

    async def evaluate_batch(
        self,
        templates: list[AttackTemplate],
        target: str = _DEFAULT_TARGET,
        payload: str = "",
    ) -> list[EvaluationResult]:
        """Evaluate a batch of templates concurrently.

        Args:
            templates: List of AttackTemplate instances to evaluate.
            target: Default {TARGET} substitution for all templates.
            payload: Default {PAYLOAD} substitution for all templates.

        Returns:
            List of EvaluationResult in the same order as templates.

        Raises:
            EvaluationError: If the entire batch fails to launch.
        """
        # Reset semaphore for each batch to avoid state leakage between calls.
        self._semaphore = asyncio.Semaphore(self._concurrency)

        tasks = [
            self._evaluate_with_limit(template, target, payload)
            for template in templates
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        logger.info(
            "Batch evaluation complete: %d/%d results obtained",
            len(results),
            len(templates),
        )
        return list(results)

    def evaluate_batch_sync(
        self,
        templates: list[AttackTemplate],
        target: str = _DEFAULT_TARGET,
        payload: str = "",
    ) -> list[EvaluationResult]:
        """Synchronous wrapper around evaluate_batch for CLI usage.

        Args:
            templates: List of AttackTemplate instances.
            target: Default {TARGET} substitution.
            payload: Default {PAYLOAD} substitution.

        Returns:
            List of EvaluationResult.
        """
        return asyncio.run(self.evaluate_batch(templates, target, payload))
