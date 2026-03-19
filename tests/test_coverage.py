"""Tests for attack surface coverage metrics computation."""

from __future__ import annotations

import pytest

from adversarial_prompt_suite.coverage import compute_coverage, format_coverage_summary
from adversarial_prompt_suite.exceptions import CoverageError
from adversarial_prompt_suite.models import (
    AttackCategory,
    ClassificationResult,
    ClassificationVerdict,
    CoverageReport,
    EvaluationResult,
)


def _make_eval(
    eval_id: str,
    template_id: str,
    category: AttackCategory,
    response: str = "I cannot help with that.",
    prompt: str = "Test prompt.",
) -> EvaluationResult:
    """Helper to build minimal EvaluationResult instances."""
    return EvaluationResult(
        id=eval_id,
        template_id=template_id,
        category=category,
        prompt=prompt,
        response=response,
        model_id="mock-model",
        latency_ms=10.0,
    )


def _make_class(
    eval_id: str,
    template_id: str,
    category: AttackCategory,
    verdict: ClassificationVerdict,
    confidence: float = 0.85,
) -> ClassificationResult:
    """Helper to build minimal ClassificationResult instances."""
    return ClassificationResult(
        evaluation_id=eval_id,
        template_id=template_id,
        category=category,
        verdict=verdict,
        confidence=confidence,
        reasoning="Test reasoning.",
        judge_used="heuristic",
    )


class TestComputeCoverage:
    """Unit tests for the compute_coverage function."""

    def test_raises_on_empty_evaluation_results(self) -> None:
        with pytest.raises(CoverageError, match="no evaluation results"):
            compute_coverage([], [], model_id="mock")

    def test_raises_on_empty_classification_results(self) -> None:
        eval_result = _make_eval("e1", "t1", AttackCategory.DIRECT_JAILBREAK)
        with pytest.raises(CoverageError, match="no classification results"):
            compute_coverage([eval_result], [], model_id="mock")

    def test_single_category_single_result(self) -> None:
        eval_result = _make_eval("e1", "t1", AttackCategory.DIRECT_JAILBREAK)
        class_result = _make_class(
            "e1", "t1", AttackCategory.DIRECT_JAILBREAK, ClassificationVerdict.SAFE
        )
        report = compute_coverage([eval_result], [class_result], model_id="mock")

        assert report.total_evaluated == 1
        assert report.total_safe == 1
        assert report.total_unsafe == 0
        assert report.total_borderline == 0

    def test_category_coverage_fraction(self) -> None:
        """With 3 of 6 categories covered, category_coverage should be 0.5."""
        categories = [
            AttackCategory.DIRECT_JAILBREAK,
            AttackCategory.PROMPT_INJECTION,
            AttackCategory.SYSTEM_EXTRACTION,
        ]
        evals = [_make_eval(f"e{i}", f"t{i}", cat) for i, cat in enumerate(categories)]
        classes = [
            _make_class(f"e{i}", f"t{i}", cat, ClassificationVerdict.SAFE)
            for i, cat in enumerate(categories)
        ]
        report = compute_coverage(evals, classes, model_id="mock")
        assert report.category_coverage == pytest.approx(3 / 6, abs=0.001)

    def test_full_category_coverage(self) -> None:
        """All 6 categories covered — coverage should be 1.0."""
        categories = list(AttackCategory)
        evals = [_make_eval(f"e{i}", f"t{i}", cat) for i, cat in enumerate(categories)]
        classes = [
            _make_class(f"e{i}", f"t{i}", cat, ClassificationVerdict.SAFE)
            for i, cat in enumerate(categories)
        ]
        report = compute_coverage(evals, classes, model_id="mock")
        assert report.category_coverage == pytest.approx(1.0)

    def test_attack_success_rate_calculation(self) -> None:
        """50% unsafe out of 4 results should give 0.5 attack success rate."""
        categories = [
            AttackCategory.DIRECT_JAILBREAK,
            AttackCategory.DIRECT_JAILBREAK,
            AttackCategory.ROLEPLAY_ESCAPE,
            AttackCategory.ROLEPLAY_ESCAPE,
        ]
        evals = [_make_eval(f"e{i}", f"t{i}", cat) for i, cat in enumerate(categories)]
        verdicts = [
            ClassificationVerdict.UNSAFE,
            ClassificationVerdict.SAFE,
            ClassificationVerdict.UNSAFE,
            ClassificationVerdict.SAFE,
        ]
        classes = [
            _make_class(f"e{i}", f"t{i}", cat, verdict)
            for i, (cat, verdict) in enumerate(zip(categories, verdicts))
        ]
        report = compute_coverage(evals, classes, model_id="mock")
        assert report.overall_attack_success_rate == pytest.approx(0.5, abs=0.001)

    def test_template_diversity_all_unique(self) -> None:
        """All unique template IDs should give diversity of 1.0."""
        categories = [AttackCategory.DIRECT_JAILBREAK] * 3
        evals = [_make_eval(f"e{i}", f"unique-template-{i}", cat) for i, cat in enumerate(categories)]
        classes = [
            _make_class(f"e{i}", f"unique-template-{i}", cat, ClassificationVerdict.SAFE)
            for i, cat in enumerate(categories)
        ]
        report = compute_coverage(evals, classes, model_id="mock")
        assert report.template_diversity == pytest.approx(1.0)

    def test_template_diversity_all_same_template(self) -> None:
        """All same template ID should give diversity < 1.0."""
        categories = [AttackCategory.DIRECT_JAILBREAK] * 4
        evals = [_make_eval(f"e{i}", "same-template", cat) for i, cat in enumerate(categories)]
        classes = [
            _make_class(f"e{i}", "same-template", cat, ClassificationVerdict.SAFE)
            for i, cat in enumerate(categories)
        ]
        report = compute_coverage(evals, classes, model_id="mock")
        assert report.template_diversity < 1.0

    def test_per_category_stats_populated(self) -> None:
        categories = list(AttackCategory)
        evals = [_make_eval(f"e{i}", f"t{i}", cat) for i, cat in enumerate(categories)]
        classes = [
            _make_class(f"e{i}", f"t{i}", cat, ClassificationVerdict.SAFE)
            for i, cat in enumerate(categories)
        ]
        report = compute_coverage(evals, classes, model_id="mock")
        assert len(report.per_category) == len(categories)

    def test_model_id_preserved_in_report(self) -> None:
        eval_result = _make_eval("e1", "t1", AttackCategory.DIRECT_JAILBREAK)
        class_result = _make_class(
            "e1", "t1", AttackCategory.DIRECT_JAILBREAK, ClassificationVerdict.SAFE
        )
        report = compute_coverage([eval_result], [class_result], model_id="my-model-v2")
        assert report.model_id == "my-model-v2"

    def test_unmatched_evaluation_ids_are_skipped(self) -> None:
        """Classification results that don't match any eval ID are ignored gracefully."""
        eval_result = _make_eval("e1", "t1", AttackCategory.DIRECT_JAILBREAK)
        class_result = _make_class(
            "WRONG-ID", "t1", AttackCategory.DIRECT_JAILBREAK, ClassificationVerdict.SAFE
        )
        # Should raise CoverageError since no evaluations can be paired.
        with pytest.raises(CoverageError):
            compute_coverage([eval_result], [class_result], model_id="mock")

    @pytest.mark.parametrize(
        "verdict,expected_field",
        [
            (ClassificationVerdict.SAFE, "total_safe"),
            (ClassificationVerdict.UNSAFE, "total_unsafe"),
            (ClassificationVerdict.BORDERLINE, "total_borderline"),
        ],
    )
    def test_verdict_counter_incremented(
        self, verdict: ClassificationVerdict, expected_field: str
    ) -> None:
        eval_result = _make_eval("e1", "t1", AttackCategory.DIRECT_JAILBREAK)
        class_result = _make_class(
            "e1", "t1", AttackCategory.DIRECT_JAILBREAK, verdict
        )
        report = compute_coverage([eval_result], [class_result], model_id="mock")
        assert getattr(report, expected_field) == 1


class TestFormatCoverageSummary:
    """Tests for the text summary formatter."""

    def test_returns_non_empty_string(self) -> None:
        eval_result = _make_eval("e1", "t1", AttackCategory.DIRECT_JAILBREAK)
        class_result = _make_class(
            "e1", "t1", AttackCategory.DIRECT_JAILBREAK, ClassificationVerdict.SAFE
        )
        report = compute_coverage([eval_result], [class_result], model_id="mock")
        summary = format_coverage_summary(report)
        assert len(summary) > 0

    def test_contains_model_id(self) -> None:
        eval_result = _make_eval("e1", "t1", AttackCategory.DIRECT_JAILBREAK)
        class_result = _make_class(
            "e1", "t1", AttackCategory.DIRECT_JAILBREAK, ClassificationVerdict.SAFE
        )
        report = compute_coverage([eval_result], [class_result], model_id="gpt-4o-test")
        summary = format_coverage_summary(report)
        assert "gpt-4o-test" in summary
