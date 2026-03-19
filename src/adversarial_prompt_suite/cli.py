"""Command-line interface for adversarial-prompt-suite.

Three sub-commands:
  run      — Evaluate a model against attack templates and produce a report.
  coverage — Print coverage metrics from an existing report JSON.
  report   — Convert a report JSON to Markdown.

Usage:
    adversarial-eval run --model gpt-4o --categories jailbreak,injection
    adversarial-eval coverage --report report.json
    adversarial-eval report --input report.json --format markdown
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from adversarial_prompt_suite.attacks import (
    load_extraction_templates,
    load_injection_templates,
    load_jailbreak_templates,
)
from adversarial_prompt_suite.classifier import SafetyClassifier
from adversarial_prompt_suite.coverage import compute_coverage, format_coverage_summary
from adversarial_prompt_suite.evaluator import Evaluator, LLMClient, MockLLMClient
from adversarial_prompt_suite.models import AttackCategory, AttackTemplate, CoverageReport
from adversarial_prompt_suite.reporter import ReportGenerator

app = typer.Typer(
    name="adversarial-eval",
    help="Systematic red-teaming framework for adversarial LLM evaluation.",
    no_args_is_help=True,
)
console = Console()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Category name to loader mapping
# ---------------------------------------------------------------------------
_CATEGORY_LOADERS: dict[str, list[str]] = {
    "jailbreak": ["DIRECT_JAILBREAK", "ROLEPLAY_ESCAPE"],
    "injection": ["PROMPT_INJECTION", "ENCODING_OBFUSCATION"],
    "extraction": ["SYSTEM_EXTRACTION", "TRAINING_DATA_EXTRACTION"],
    "all": list(c.value for c in AttackCategory),
}


def _load_templates_by_category(categories_str: str) -> list[AttackTemplate]:
    """Load attack templates for the specified categories.

    Args:
        categories_str: Comma-separated category names from _CATEGORY_LOADERS.

    Returns:
        Deduplicated list of AttackTemplate instances.

    Raises:
        typer.BadParameter: If an unknown category name is given.
    """
    requested = [c.strip().lower() for c in categories_str.split(",")]
    target_categories: set[str] = set()

    for name in requested:
        if name not in _CATEGORY_LOADERS:
            valid = ", ".join(_CATEGORY_LOADERS.keys())
            raise typer.BadParameter(
                f"Unknown category '{name}'. Valid options: {valid}"
            )
        target_categories.update(_CATEGORY_LOADERS[name])

    all_templates = (
        load_jailbreak_templates()
        + load_injection_templates()
        + load_extraction_templates()
    )

    return [t for t in all_templates if t.category.value in target_categories]


def _setup_logging(verbose: bool) -> None:
    """Configure root logger based on verbosity flag.

    Args:
        verbose: If True, set DEBUG level; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command("run")
def run_evaluation(
    model: str = typer.Option("mock", help="Model ID (or 'mock' for offline testing)."),
    categories: str = typer.Option(
        "all", help="Comma-separated categories: jailbreak, injection, extraction, all."
    ),
    output: Path = typer.Option(Path("report.json"), help="Output JSON report path."),
    target: str = typer.Option(
        "explain how to do something harmful",
        help="Default {TARGET} substitution for prompt templates.",
    ),
    api_key: Optional[str] = typer.Option(None, envvar="OPENAI_API_KEY", help="OpenAI API key."),
    base_url: str = typer.Option("https://api.openai.com/v1", help="OpenAI-compatible base URL."),
    concurrency: int = typer.Option(5, help="Concurrent evaluation requests."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Evaluate a model against adversarial attack templates.

    Runs all selected attack categories against the specified model,
    classifies responses, computes coverage, and writes a JSON report.
    """
    _setup_logging(verbose)

    templates = _load_templates_by_category(categories)
    console.print(
        f"[bold]Loaded [cyan]{len(templates)}[/cyan] templates across "
        f"[cyan]{len(set(t.category for t in templates))}[/cyan] categories[/bold]"
    )

    # Initialise the LLM client.
    if model == "mock" or not api_key:
        console.print("[yellow]Using mock LLM client (no API key provided).[/yellow]")
        client = MockLLMClient(model_id=model)
    else:
        client = LLMClient(
            base_url=base_url,
            api_key=api_key,
            model_id=model,
        )

    evaluator = Evaluator(client=client, concurrency=concurrency)
    classifier = SafetyClassifier()
    generator = ReportGenerator()

    console.print(f"[bold]Evaluating model:[/bold] {model}")

    with console.status("Running evaluations..."):
        eval_results = asyncio.run(evaluator.evaluate_batch(templates, target=target))

    console.print(f"[green]Evaluations complete:[/green] {len(eval_results)} results")

    with console.status("Classifying responses..."):
        class_results = classifier.classify_batch(eval_results)

    console.print(f"[green]Classifications complete:[/green] {len(class_results)} verdicts")

    report = compute_coverage(eval_results, class_results, model_id=model)

    generator.generate_json(report, eval_results, class_results, output)
    console.print(f"[green]Report written to:[/green] {output}")

    generator.print_summary(report)


@app.command("coverage")
def show_coverage(
    report: Path = typer.Argument(..., help="Path to report.json from 'run' command."),
) -> None:
    """Print attack surface coverage metrics from an existing report."""
    if not report.exists():
        console.print(f"[red]Report file not found:[/red] {report}")
        raise typer.Exit(1)

    try:
        with report.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        coverage_report = CoverageReport.model_validate(data["coverage"])
    except Exception as exc:
        console.print(f"[red]Failed to load report:[/red] {exc}")
        raise typer.Exit(1)

    table = Table(title=f"Coverage: {coverage_report.model_id}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Evaluated", str(coverage_report.total_evaluated))
    table.add_row("UNSAFE", f"{coverage_report.total_unsafe} ({coverage_report.overall_attack_success_rate:.1%})")
    table.add_row("SAFE", str(coverage_report.total_safe))
    table.add_row("BORDERLINE", str(coverage_report.total_borderline))
    table.add_row("Category Coverage", f"{coverage_report.category_coverage:.1%}")
    table.add_row("Template Diversity", f"{coverage_report.template_diversity:.1%}")
    table.add_row("False Refusal Rate", f"{coverage_report.false_refusal_rate:.1%}")

    console.print(table)


@app.command("report")
def generate_report(
    input_path: Path = typer.Option(..., "--input", help="Path to report.json from 'run' command."),
    format: str = typer.Option("markdown", help="Output format: markdown or json."),
    output: Optional[Path] = typer.Option(None, help="Output path (defaults to report.md)."),
) -> None:
    """Convert a JSON report to the requested output format."""
    if not input_path.exists():
        console.print(f"[red]Input file not found:[/red] {input_path}")
        raise typer.Exit(1)

    try:
        with input_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        coverage_report = CoverageReport.model_validate(data["coverage"])
    except Exception as exc:
        console.print(f"[red]Failed to load input:[/red] {exc}")
        raise typer.Exit(1)

    if format.lower() == "markdown":
        out_path = output or input_path.with_suffix(".md")
        from adversarial_prompt_suite.models import EvaluationResult, ClassificationResult

        eval_results = [EvaluationResult.model_validate(e) for e in data.get("evaluations", [])]
        class_results = [ClassificationResult.model_validate(c) for c in data.get("classifications", [])]

        generator = ReportGenerator()
        generator.generate_markdown(coverage_report, eval_results, class_results, out_path)
        console.print(f"[green]Markdown report written to:[/green] {out_path}")
    else:
        console.print(f"[red]Unknown format:[/red] {format}. Use 'markdown'.")
        raise typer.Exit(1)


def main() -> None:
    """Entry point for the adversarial-eval CLI."""
    app()


if __name__ == "__main__":
    main()
