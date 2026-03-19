# Architecture: adversarial-prompt-suite

## Overview

The framework is a five-stage pipeline: attack templates flow through an async evaluator,
responses are classified by a two-layer safety classifier, coverage metrics are computed,
and a structured report is emitted.

## Component Responsibilities

| Component | File | Responsibility |
|-----------|------|----------------|
| Attack Loaders | `attacks/jailbreak.py`, `injection.py`, `extraction.py` | Provide typed AttackTemplate instances per category |
| Evaluator | `evaluator.py` | Async batch evaluation against any OpenAI-compatible endpoint |
| Classifier | `classifier.py` | Two-layer safety classification (heuristic + LLM judge) |
| Coverage | `coverage.py` | Compute attack surface coverage metrics across 6 categories |
| Reporter | `reporter.py` | Emit JSON + Markdown reports |
| Dataset | `dataset.py` | JSONL read/write for all pipeline artifacts |
| CLI | `cli.py` | `run`, `coverage`, `report` sub-commands via typer |

## Data Flow

```
Attack Templates (JSONL + built-in)
    |
    v
Evaluator (asyncio, semaphore-gated concurrency)
    |  --> LLM API (OpenAI-compatible)
    v
EvaluationResult[] (prompt, response, latency, category)
    |
    v
SafetyClassifier
    |-- HeuristicClassifier (regex, O(n), zero-cost)
    |   --> SAFE / UNSAFE (high confidence cases)
    |-- LLMJudgeClassifier (escalation only for BORDERLINE)
    v
ClassificationResult[] (verdict, confidence, reasoning)
    |
    v
compute_coverage()
    |-- category_coverage   (fraction of 6 categories tested)
    |-- template_diversity  (unique templates / total attempts)
    |-- attack_success_rate (per category and overall)
    |-- false_refusal_rate  (benign prompts incorrectly refused)
    v
CoverageReport
    |
    v
ReportGenerator
    |-- JSON (machine-readable, CI integration)
    |-- Markdown (human-readable, PR comments)
```

## Attack Category Taxonomy

| Category | Attack Mechanism |
|----------|-----------------|
| DIRECT_JAILBREAK | Instruction override via imperative framing |
| ROLEPLAY_ESCAPE | Character persona that claims no restrictions |
| PROMPT_INJECTION | Instructions embedded in user-controlled data fields |
| SYSTEM_EXTRACTION | Eliciting system prompt contents |
| TRAINING_DATA_EXTRACTION | Verbatim memorization probes |
| ENCODING_OBFUSCATION | Token-level filter bypass via encoding |

## Classifier Design

The heuristic layer handles ~80% of responses using compiled regex patterns.
BORDERLINE cases are escalated to the LLM judge, which sends a structured
classification prompt to a small, fast model (gpt-4o-mini by default) and
parses the JSON verdict. This design minimises judge API cost while maintaining
accuracy on edge cases.

## Concurrency Model

The evaluator uses `asyncio.gather` with a `asyncio.Semaphore` to gate
concurrent requests. This avoids rate-limit errors while achieving near-linear
throughput scaling up to the concurrency limit. The semaphore is reset between
`evaluate_batch` calls to prevent stale state from accumulated timeouts.
