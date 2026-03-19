"""Pydantic data models for adversarial-prompt-suite.

All pipeline components exchange these typed structures, ensuring
schema consistency across attacks, evaluation, classification, and reporting.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class AttackCategory(str, Enum):
    """Taxonomy of adversarial attack categories.

    These six categories cover the primary attack surfaces identified in
    red-teaming literature. Each maps to a distinct evasion mechanism.
    """

    DIRECT_JAILBREAK = "DIRECT_JAILBREAK"
    """Explicit instruction override — DAN-style 'ignore all previous instructions'."""

    ROLEPLAY_ESCAPE = "ROLEPLAY_ESCAPE"
    """Character or persona roleplay that grants fictional license to bypass safety."""

    PROMPT_INJECTION = "PROMPT_INJECTION"
    """Injecting instructions through user-controlled content fields."""

    SYSTEM_EXTRACTION = "SYSTEM_EXTRACTION"
    """Attempts to reveal system prompt contents through indirect elicitation."""

    TRAINING_DATA_EXTRACTION = "TRAINING_DATA_EXTRACTION"
    """Verbatim memorization probes targeting training corpus regurgitation."""

    ENCODING_OBFUSCATION = "ENCODING_OBFUSCATION"
    """Unicode homoglyphs, base64, leetspeak, or other encoding-layer bypasses."""


class ClassificationVerdict(str, Enum):
    """Three-class output from the safety classifier.

    BORDERLINE cases are escalated to the LLM judge layer to avoid
    false positives from rule-based heuristics.
    """

    SAFE = "SAFE"
    UNSAFE = "UNSAFE"
    BORDERLINE = "BORDERLINE"


class AttackTemplate(BaseModel):
    """A single adversarial prompt template with metadata.

    Templates use {TARGET} and {PAYLOAD} placeholders that are substituted
    at evaluation time. This enables reuse across different target behaviors.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    category: AttackCategory
    name: str = Field(..., min_length=1, max_length=120)
    template: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    severity: int = Field(..., ge=1, le=5)
    tags: list[str] = Field(default_factory=list)
    source: Optional[str] = None

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, value: int) -> int:
        """Ensure severity is in the 1-5 range.

        Args:
            value: Proposed severity integer.

        Returns:
            Validated severity value.

        Raises:
            ValueError: If severity is outside [1, 5].
        """
        if not 1 <= value <= 5:
            raise ValueError(f"severity must be between 1 and 5, got {value}")
        return value

    model_config = {"frozen": True}


class EvaluationResult(BaseModel):
    """Output from evaluating a single attack template against an LLM.

    Captures the full request-response pair plus metadata needed for
    downstream classification and coverage analysis.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    template_id: str
    category: AttackCategory
    prompt: str
    response: str
    model_id: str
    latency_ms: float = Field(..., ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error: Optional[str] = None

    model_config = {"frozen": True}


class ClassificationResult(BaseModel):
    """Safety verdict for a single EvaluationResult.

    The reasoning field captures the classifier's decision chain,
    enabling audit and downstream improvement of the classifier itself.
    """

    evaluation_id: str
    template_id: str
    category: AttackCategory
    verdict: ClassificationVerdict
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    judge_used: str = Field(default="heuristic")

    model_config = {"frozen": True}


class CategoryStats(BaseModel):
    """Per-category aggregated statistics within a coverage report."""

    category: AttackCategory
    total_attempts: int = Field(..., ge=0)
    successful_attacks: int = Field(..., ge=0)
    false_refusals: int = Field(..., ge=0)
    attack_success_rate: float = Field(..., ge=0.0, le=1.0)
    false_refusal_rate: float = Field(..., ge=0.0, le=1.0)

    model_config = {"frozen": True}


class CoverageReport(BaseModel):
    """Aggregate attack surface coverage report for a full evaluation run.

    The coverage metrics answer the question: "what fraction of the known
    attack surface did this evaluation exercise?"
    """

    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_prompts: int = Field(..., ge=0)
    total_evaluated: int = Field(..., ge=0)
    total_unsafe: int = Field(..., ge=0)
    total_safe: int = Field(..., ge=0)
    total_borderline: int = Field(..., ge=0)
    category_coverage: float = Field(..., ge=0.0, le=1.0)
    """Fraction of the 6 attack categories tested."""
    template_diversity: float = Field(..., ge=0.0, le=1.0)
    """Unique prompt structures / total attempts."""
    overall_attack_success_rate: float = Field(..., ge=0.0, le=1.0)
    false_refusal_rate: float = Field(..., ge=0.0, le=1.0)
    per_category: list[CategoryStats] = Field(default_factory=list)
    evaluation_ids: list[str] = Field(default_factory=list)

    model_config = {"frozen": True}
