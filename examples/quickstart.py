"""Quickstart example: full adversarial evaluation pipeline in 30 lines.

This script demonstrates the complete pipeline without making any real
API calls. It uses the built-in MockLLMClient so it runs offline.

Run with:
    python examples/quickstart.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

# Allow running as a script without installation.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adversarial_prompt_suite.attacks import load_jailbreak_templates
from adversarial_prompt_suite.classifier import SafetyClassifier
from adversarial_prompt_suite.coverage import compute_coverage, format_coverage_summary
from adversarial_prompt_suite.evaluator import Evaluator, MockLLMClient

logging.basicConfig(level=logging.WARNING)


async def main() -> None:
    # Step 1: Load attack templates for jailbreak categories.
    templates = load_jailbreak_templates()
    print(f"Loaded {len(templates)} jailbreak attack templates.")

    # Step 2: Run evaluation with the mock LLM client (no API key required).
    client = MockLLMClient(model_id="demo-model")
    evaluator = Evaluator(client=client, concurrency=5)
    eval_results = await evaluator.evaluate_batch(templates)
    print(f"Evaluated {len(eval_results)} prompts.")

    # Step 3: Classify responses.
    classifier = SafetyClassifier()  # No LLM judge configured — uses heuristics only.
    classification_results = classifier.classify_batch(eval_results)

    verdict_counts = {}
    for c in classification_results:
        verdict_counts[c.verdict.value] = verdict_counts.get(c.verdict.value, 0) + 1
    print(f"Classification results: {verdict_counts}")

    # Step 4: Compute attack surface coverage.
    report = compute_coverage(eval_results, classification_results, model_id="demo-model")

    # Step 5: Print coverage summary.
    print("\n" + format_coverage_summary(report))
    print("\nQuickstart complete.")


if __name__ == "__main__":
    asyncio.run(main())
