"""Statistics helpers for the defense-eval experiment.

Computes per (category, condition) attack success rates with Wilson 95%
confidence intervals, benign refusal rates, and renders Markdown tables.

Pure functions: they operate on plain dicts read from the JSONL outputs of
experiments/defense_eval/run.py, so they need no model or network access.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from adversarial_prompt_suite.models import AttackCategory

# Display names keep tables readable while tests can still find the raw
# category values (e.g. "DIRECT_JAILBREAK") in the rendered Markdown.
_CATEGORY_LABELS = {
    "DIRECT_JAILBREAK": "Direct jailbreak",
    "ROLEPLAY_ESCAPE": "Roleplay escape",
    "PROMPT_INJECTION": "Prompt injection",
    "SYSTEM_EXTRACTION": "System extraction",
    "TRAINING_DATA_EXTRACTION": "Training-data extraction",
    "ENCODING_OBFUSCATION": "Encoding obfuscation",
}

_CONDITION_ORDER = ["baseline", "sysprompt", "outfilter", "verifygate"]


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Args:
        successes: Number of observed successes.
        n: Number of trials.
        z: Normal quantile (1.96 gives a 95% interval).

    Returns:
        (lower, upper) bounds of the interval, both floats in [0, 1].
    """
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half_width = (
        z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    )
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def summarize_attack_rows(
    rows: Iterable[dict],
) -> dict[tuple[str, str], dict[str, float | int]]:
    """Aggregate attack rows into per (category, condition) stats.

    Errored evaluations are counted in ``errors`` and excluded from ``n``,
    so a flaky API cannot silently inflate the safe count.

    Args:
        rows: Dicts as written to rows.jsonl by run.py.

    Returns:
        Dict keyed (category, condition) with n, unsafe, borderline, safe,
        errors, asr, asr_lo, asr_hi, borderline_rate.
    """
    buckets: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (row["category"], row["condition"])
        buckets.setdefault(key, []).append(row)

    summary: dict[tuple[str, str], dict[str, float | int]] = {}
    for key, cell in buckets.items():
        errors = sum(1 for r in cell if r.get("error"))
        evaluated = [r for r in cell if not r.get("error")]
        n = len(evaluated)
        unsafe = sum(1 for r in evaluated if r.get("verdict") == "UNSAFE")
        borderline = sum(1 for r in evaluated if r.get("verdict") == "BORDERLINE")
        safe = n - unsafe - borderline
        asr = unsafe / n if n else 0.0
        asr_lo, asr_hi = wilson(unsafe, n)
        summary[key] = {
            "n": n,
            "unsafe": unsafe,
            "borderline": borderline,
            "safe": safe,
            "errors": errors,
            "asr": asr,
            "asr_lo": asr_lo,
            "asr_hi": asr_hi,
            "borderline_rate": borderline / n if n else 0.0,
        }
    return summary


def summarize_benign_rows(
    rows: Iterable[dict],
) -> dict[str, dict[str, float | int]]:
    """Aggregate benign rows into per-condition refusal stats.

    Args:
        rows: Dicts as written to benign.jsonl by run.py.

    Returns:
        Dict keyed condition with n, refused, refusal_rate.
    """
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(row["condition"], []).append(row)

    summary: dict[str, dict[str, float | int]] = {}
    for condition, cell in buckets.items():
        n = len(cell)
        refused = sum(1 for r in cell if r.get("refused"))
        summary[condition] = {
            "n": n,
            "refused": refused,
            "refusal_rate": refused / n if n else 0.0,
        }
    return summary


def _pct(value: float) -> str:
    """Format a 0-1 rate as a percentage with one decimal."""
    return f"{100.0 * value:.1f}%"


def _ordered_conditions(
    attack_summary: dict[tuple[str, str], dict[str, float | int]],
) -> list[str]:
    """Conditions in canonical order, then any extras seen in the data."""
    seen: list[str] = []
    for (_, condition) in attack_summary:
        if condition not in seen:
            seen.append(condition)
    ordered = [c for c in _CONDITION_ORDER if c in seen]
    ordered.extend(c for c in seen if c not in ordered)
    return ordered


def render_markdown(
    attack_summary: dict[tuple[str, str], dict[str, float | int]],
    benign_summary: dict[str, dict[str, float | int]],
    manifest: dict,
) -> str:
    """Render the experiment summary as Markdown.

    One table per attack category (rows = conditions, columns = n, ASR,
    95% CI, delta vs baseline in percentage points), then the benign
    refusal table, then a methods block.

    Args:
        attack_summary: Output of summarize_attack_rows.
        benign_summary: Output of summarize_benign_rows.
        manifest: The run manifest dict (models, replicates, judge, date).

    Returns:
        Markdown string. No emojis.
    """
    conditions = _ordered_conditions(attack_summary)
    lines: list[str] = ["# Defense evaluation results", ""]

    for category in AttackCategory:
        label = _CATEGORY_LABELS.get(category.value, category.value)
        lines.append(f"## {label} ({category.value})")
        lines.append("")
        lines.append("| condition | n | ASR | 95% CI | delta vs baseline (pp) |")
        lines.append("| --- | ---: | ---: | --- | ---: |")
        baseline = attack_summary.get((category.value, "baseline"))
        baseline_asr = float(baseline["asr"]) if baseline else 0.0
        for condition in conditions:
            cell = attack_summary.get((category.value, condition))
            if cell is None:
                lines.append(
                    f"| {condition} | - | - | - | - |"
                )
                continue
            asr = float(cell["asr"])
            delta = (
                f"{100.0 * (asr - baseline_asr):+.1f}"
                if condition != "baseline" and baseline is not None
                else "-"
            )
            lines.append(
                f"| {condition} | {cell['n']} | {_pct(asr)} | "
                f"({_pct(float(cell['asr_lo']))}, {_pct(float(cell['asr_hi']))}) | "
                f"{delta} |"
            )
        lines.append("")

    lines.append("## Benign refusal rate")
    lines.append("")
    lines.append(
        "Refusal detected by a simple case-insensitive substring match on "
        "a small refusal-phrase list (see run.py). No LLM judge is used here."
    )
    lines.append("")
    lines.append("| condition | n | refused | refusal rate |")
    lines.append("| --- | ---: | ---: | ---: |")
    benign_order = [c for c in _CONDITION_ORDER if c in benign_summary]
    benign_order.extend(c for c in benign_summary if c not in benign_order)
    for condition in benign_order:
        cell = benign_summary[condition]
        lines.append(
            f"| {condition} | {cell['n']} | {cell['refused']} | "
            f"{_pct(float(cell['refusal_rate']))} |"
        )
    lines.append("")

    lines.append("## Methods")
    lines.append("")
    lines.append(f"- Target model: {manifest.get('target_model', 'unknown')}")
    lines.append(f"- Judge model: {manifest.get('judge_model', 'unknown')}")
    lines.append(f"- Replicates: {manifest.get('replicates', 'unknown')}")
    lines.append(
        "- Judge method: heuristic classifier first (refusal/compliance regex), "
        "LLM judge only for borderline cases. Defense rows record the real "
        "judge model id, not the adapter placeholder."
    )
    lines.append(
        f"- Run started: {manifest.get('started_utc', 'unknown')} "
        f"(run id {manifest.get('run_id', 'unknown')})"
    )
    lines.append(
        "- Replicates are exact repeats at temperature 0; they capture API "
        "and judge nondeterminism, not prompt diversity."
    )
    lines.append("")
    return "\n".join(lines)
