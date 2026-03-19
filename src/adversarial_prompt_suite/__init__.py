"""
adversarial-prompt-suite — Systematic red-teaming framework for LLM adversarial evaluation.

Provides structured attack templates, evaluation pipelines, safety classification,
attack surface coverage metrics, and structured reporting.
"""

from adversarial_prompt_suite.models import (
    AttackCategory,
    AttackTemplate,
    EvaluationResult,
    ClassificationVerdict,
    ClassificationResult,
    CoverageReport,
)

__version__ = "0.1.0"
__author__ = "Rajath John"
__email__ = "jrajath94@gmail.com"

__all__ = [
    "AttackCategory",
    "AttackTemplate",
    "EvaluationResult",
    "ClassificationVerdict",
    "ClassificationResult",
    "CoverageReport",
]
