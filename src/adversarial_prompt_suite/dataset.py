"""Dataset I/O for adversarial attack templates and evaluation results.

All persistence uses JSONL (one JSON object per line) for streaming-friendly
reads, easy grep-ability, and git-diff-ability. Pydantic handles schema
validation on load.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Union

from adversarial_prompt_suite.exceptions import DatasetError
from adversarial_prompt_suite.models import (
    AttackTemplate,
    ClassificationResult,
    CoverageReport,
    EvaluationResult,
)

logger = logging.getLogger(__name__)

# Type alias for any serialisable model in the pipeline.
_PipelineRecord = Union[
    AttackTemplate, EvaluationResult, ClassificationResult, CoverageReport
]


def _load_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of parsed JSON objects.

    Raises:
        DatasetError: If the file does not exist or any line is invalid JSON.
    """
    if not path.exists():
        raise DatasetError(f"Dataset file not found: {path}")

    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise DatasetError(
                    f"Invalid JSON on line {line_number} of {path}: {exc}"
                ) from exc

    logger.debug("Loaded %d records from %s", len(records), path)
    return records


def _write_jsonl(records: list[dict], path: Path) -> None:
    """Write a list of dicts to a JSONL file, creating parent dirs as needed.

    Args:
        records: List of JSON-serialisable dicts.
        path: Destination file path.

    Raises:
        DatasetError: If the file cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record) + "\n")
    except OSError as exc:
        raise DatasetError(f"Failed to write dataset to {path}: {exc}") from exc

    logger.debug("Wrote %d records to %s", len(records), path)


def load_attack_templates(path: Path) -> list[AttackTemplate]:
    """Load attack templates from a JSONL file.

    Args:
        path: Path to the JSONL file containing AttackTemplate records.

    Returns:
        List of validated AttackTemplate instances.

    Raises:
        DatasetError: If the file cannot be read or any record is invalid.
    """
    raw_records = _load_jsonl(path)
    templates: list[AttackTemplate] = []
    for i, record in enumerate(raw_records):
        try:
            templates.append(AttackTemplate.model_validate(record))
        except Exception as exc:
            raise DatasetError(
                f"Invalid AttackTemplate at record {i} in {path}: {exc}"
            ) from exc

    logger.info("Loaded %d attack templates from %s", len(templates), path)
    return templates


def save_attack_templates(templates: list[AttackTemplate], path: Path) -> None:
    """Persist a list of AttackTemplate instances to a JSONL file.

    Args:
        templates: List of AttackTemplate instances to save.
        path: Destination JSONL file path.

    Raises:
        DatasetError: If the file cannot be written.
    """
    records = [t.model_dump(mode="json") for t in templates]
    _write_jsonl(records, path)
    logger.info("Saved %d attack templates to %s", len(templates), path)


def load_evaluation_results(path: Path) -> list[EvaluationResult]:
    """Load evaluation results from a JSONL file.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of validated EvaluationResult instances.

    Raises:
        DatasetError: If the file cannot be read or any record is invalid.
    """
    raw_records = _load_jsonl(path)
    results: list[EvaluationResult] = []
    for i, record in enumerate(raw_records):
        try:
            results.append(EvaluationResult.model_validate(record))
        except Exception as exc:
            raise DatasetError(
                f"Invalid EvaluationResult at record {i} in {path}: {exc}"
            ) from exc

    logger.info("Loaded %d evaluation results from %s", len(results), path)
    return results


def save_evaluation_results(results: list[EvaluationResult], path: Path) -> None:
    """Persist evaluation results to a JSONL file.

    Args:
        results: List of EvaluationResult instances.
        path: Destination JSONL file path.

    Raises:
        DatasetError: If the file cannot be written.
    """
    records = [r.model_dump(mode="json") for r in results]
    _write_jsonl(records, path)
    logger.info("Saved %d evaluation results to %s", len(results), path)


def save_coverage_report(report: CoverageReport, path: Path) -> None:
    """Persist a CoverageReport as a single JSON file.

    Args:
        report: CoverageReport instance to save.
        path: Destination JSON file path (not JSONL — one report per file).

    Raises:
        DatasetError: If the file cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(report.model_dump(mode="json"), fh, indent=2, default=str)
    except OSError as exc:
        raise DatasetError(f"Failed to write coverage report to {path}: {exc}") from exc

    logger.info("Saved coverage report to %s", path)


def load_coverage_report(path: Path) -> CoverageReport:
    """Load a CoverageReport from a JSON file.

    Args:
        path: Path to the JSON coverage report file.

    Returns:
        Validated CoverageReport instance.

    Raises:
        DatasetError: If the file is missing or invalid.
    """
    if not path.exists():
        raise DatasetError(f"Coverage report not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return CoverageReport.model_validate(raw)
    except Exception as exc:
        raise DatasetError(f"Failed to load coverage report from {path}: {exc}") from exc
