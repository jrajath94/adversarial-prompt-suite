"""Report generator for adversarial evaluation runs.

Produces two output formats:
  - JSON: Structured output for programmatic consumption / CI integration.
  - Markdown: Human-readable report for team review.

Both formats are derived from CoverageReport, which is the canonical
output of a complete evaluation + classification + coverage pipeline run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from adversarial_prompt_suite.coverage import _TOTAL_CATEGORIES, format_coverage_summary
from adversarial_prompt_suite.exceptions import ReportGenerationError
from adversarial_prompt_suite.models import (
    ClassificationResult,
    ClassificationVerdict,
    CoverageReport,
    EvaluationResult,
)

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates structured reports from evaluation pipeline outputs.

    Usage:
        generator = ReportGenerator()
        generator.generate_json(report, eval_results, class_results, output_path)
        generator.generate_markdown(report, eval_results, class_results, output_path)
    """

    def generate_json(
        self,
        report: CoverageReport,
        evaluation_results: list[EvaluationResult],
        classification_results: list[ClassificationResult],
        output_path: Path,
    ) -> None:
        """Write a JSON report combining coverage, evaluations, and classifications.

        Args:
            report: CoverageReport with aggregate metrics.
            evaluation_results: Raw evaluation outputs.
            classification_results: Safety verdict per evaluation.
            output_path: Destination JSON file path.

        Raises:
            ReportGenerationError: If serialisation or file I/O fails.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            payload = {
                "coverage": report.model_dump(mode="json"),
                "evaluations": [
                    e.model_dump(mode="json") for e in evaluation_results
                ],
                "classifications": [
                    c.model_dump(mode="json") for c in classification_results
                ],
            }
            with output_path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, default=str)
        except Exception as exc:
            raise ReportGenerationError(
                f"Failed to write JSON report to {output_path}: {exc}"
            ) from exc

        logger.info("JSON report written to %s", output_path)

    def generate_markdown(
        self,
        report: CoverageReport,
        evaluation_results: list[EvaluationResult],
        classification_results: list[ClassificationResult],
        output_path: Path,
    ) -> None:
        """Write a Markdown report suitable for GitHub PR comments or team review.

        Args:
            report: CoverageReport with aggregate metrics.
            evaluation_results: Raw evaluation outputs.
            classification_results: Safety verdict per evaluation.
            output_path: Destination Markdown file path.

        Raises:
            ReportGenerationError: If serialisation or file I/O fails.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            md = self._build_markdown(report, evaluation_results, classification_results)
            with output_path.open("w", encoding="utf-8") as fh:
                fh.write(md)
        except ReportGenerationError:
            raise
        except Exception as exc:
            raise ReportGenerationError(
                f"Failed to write Markdown report to {output_path}: {exc}"
            ) from exc

        logger.info("Markdown report written to %s", output_path)

    def _build_markdown(
        self,
        report: CoverageReport,
        evaluation_results: list[EvaluationResult],
        classification_results: list[ClassificationResult],
    ) -> str:
        """Construct the Markdown report string.

        Args:
            report: CoverageReport with aggregate metrics.
            evaluation_results: Raw evaluation outputs.
            classification_results: Safety verdict per evaluation.

        Returns:
            Markdown string.
        """
        class_index = {c.evaluation_id: c for c in classification_results}

        sections: list[str] = [
            f"# Adversarial Evaluation Report\n",
            f"**Model:** `{report.model_id}`  ",
            f"**Generated:** {report.generated_at.strftime('%Y-%m-%d %H:%M UTC')}  ",
            f"**Run ID:** `{report.run_id}`\n",
            "---\n",
            "## Summary\n",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Total Prompts | {report.total_evaluated} |",
            f"| UNSAFE | {report.total_unsafe} ({report.overall_attack_success_rate:.1%}) |",
            f"| SAFE | {report.total_safe} |",
            f"| BORDERLINE | {report.total_borderline} |",
            f"| Category Coverage | {report.category_coverage:.1%} ({int(report.category_coverage * _TOTAL_CATEGORIES)}/{_TOTAL_CATEGORIES} categories) |",
            f"| Template Diversity | {report.template_diversity:.1%} |",
            f"| False Refusal Rate | {report.false_refusal_rate:.1%} |",
            "\n---\n",
            "## Per-Category Breakdown\n",
            "| Category | Attempts | Successful Attacks | Success Rate |",
            "| --- | --- | --- | --- |",
        ]

        for stats in report.per_category:
            sections.append(
                f"| `{stats.category.value}` | {stats.total_attempts} | "
                f"{stats.successful_attacks} | {stats.attack_success_rate:.1%} |"
            )

        sections.extend([
            "\n---\n",
            "## Unsafe Responses\n",
            "> Showing top 10 UNSAFE classifications.\n",
        ])

        unsafe_pairs = [
            (e, class_index[e.id])
            for e in evaluation_results
            if e.id in class_index
            and class_index[e.id].verdict == ClassificationVerdict.UNSAFE
        ][:10]

        if not unsafe_pairs:
            sections.append("_No UNSAFE responses found._\n")
        else:
            for eval_result, classification in unsafe_pairs:
                sections.extend([
                    f"### `{eval_result.category.value}` — Template `{eval_result.template_id[:8]}`\n",
                    f"**Confidence:** {classification.confidence:.0%}  ",
                    f"**Reasoning:** {classification.reasoning}\n",
                    f"<details><summary>Prompt (click to expand)</summary>\n\n"
                    f"```\n{eval_result.prompt[:500]}\n```\n</details>\n",
                    f"<details><summary>Response (click to expand)</summary>\n\n"
                    f"```\n{eval_result.response[:500]}\n```\n</details>\n",
                ])

        return "\n".join(sections) + "\n"

    def print_summary(self, report: CoverageReport) -> None:
        """Print a text summary of the coverage report to stdout.

        Args:
            report: CoverageReport to summarise.
        """
        print(format_coverage_summary(report))
