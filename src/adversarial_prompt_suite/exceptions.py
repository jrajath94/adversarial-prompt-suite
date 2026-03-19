"""Custom exception hierarchy for adversarial-prompt-suite.

All exceptions inherit from AdversarialSuiteError to allow broad catches
while still enabling targeted handling of specific failure modes.
"""


class AdversarialSuiteError(Exception):
    """Base exception for all adversarial-prompt-suite errors."""


class AttackLoadError(AdversarialSuiteError):
    """Raised when attack templates cannot be loaded or parsed.

    Common causes:
    - JSONL file not found or malformed
    - Required fields missing from a template record
    - Invalid attack category value
    """


class EvaluationError(AdversarialSuiteError):
    """Raised when an evaluation run fails to complete.

    Common causes:
    - LLM API connection failure
    - Rate limiting or quota exhaustion
    - Unexpected API response structure
    """


class ClassificationError(AdversarialSuiteError):
    """Raised when a response cannot be classified.

    Common causes:
    - LLM judge returns malformed output
    - Classification rules encounter an unexpected response format
    """


class CoverageError(AdversarialSuiteError):
    """Raised when coverage metrics cannot be computed.

    Common causes:
    - Empty evaluation result set
    - Required fields missing from EvaluationResult objects
    """


class ReportGenerationError(AdversarialSuiteError):
    """Raised when a report cannot be generated or serialized."""


class DatasetError(AdversarialSuiteError):
    """Raised when dataset read/write operations fail."""
