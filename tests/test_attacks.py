"""Tests for attack template loaders."""

from __future__ import annotations

import pytest

from adversarial_prompt_suite.attacks.extraction import load_extraction_templates
from adversarial_prompt_suite.attacks.injection import load_injection_templates
from adversarial_prompt_suite.attacks.jailbreak import load_jailbreak_templates
from adversarial_prompt_suite.models import AttackCategory, AttackTemplate


class TestJailbreakTemplates:
    """Tests for DIRECT_JAILBREAK and ROLEPLAY_ESCAPE template loaders."""

    def test_load_returns_non_empty_list(self) -> None:
        templates = load_jailbreak_templates()
        assert len(templates) > 0

    def test_all_items_are_attack_templates(self) -> None:
        templates = load_jailbreak_templates()
        assert all(isinstance(t, AttackTemplate) for t in templates)

    def test_contains_both_categories(self) -> None:
        templates = load_jailbreak_templates()
        categories = {t.category for t in templates}
        assert AttackCategory.DIRECT_JAILBREAK in categories
        assert AttackCategory.ROLEPLAY_ESCAPE in categories

    def test_no_other_categories_present(self) -> None:
        templates = load_jailbreak_templates()
        valid = {AttackCategory.DIRECT_JAILBREAK, AttackCategory.ROLEPLAY_ESCAPE}
        for t in templates:
            assert t.category in valid, f"Unexpected category: {t.category}"

    @pytest.mark.parametrize("severity", [1, 2, 3, 4, 5])
    def test_severity_valid_range_accepted(self, severity: int) -> None:
        """Template creation should accept all valid severity values."""
        template = AttackTemplate(
            category=AttackCategory.DIRECT_JAILBREAK,
            name="test",
            template="Test {TARGET}",
            description="test",
            severity=severity,
        )
        assert template.severity == severity

    def test_severity_out_of_range_raises(self) -> None:
        with pytest.raises(Exception):
            AttackTemplate(
                category=AttackCategory.DIRECT_JAILBREAK,
                name="test",
                template="Test",
                description="test",
                severity=6,
            )

    def test_all_templates_have_non_empty_names(self) -> None:
        templates = load_jailbreak_templates()
        for t in templates:
            assert t.name.strip(), f"Template has empty name: {t.id}"

    def test_all_templates_have_non_empty_descriptions(self) -> None:
        templates = load_jailbreak_templates()
        for t in templates:
            assert t.description.strip(), f"Template has empty description: {t.id}"

    def test_template_ids_are_unique(self) -> None:
        templates = load_jailbreak_templates()
        ids = [t.id for t in templates]
        assert len(ids) == len(set(ids)), "Duplicate template IDs found."

    def test_direct_jailbreak_count(self) -> None:
        """Verify we have at least 8 direct jailbreak templates."""
        templates = load_jailbreak_templates()
        direct = [t for t in templates if t.category == AttackCategory.DIRECT_JAILBREAK]
        assert len(direct) >= 8

    def test_roleplay_escape_count(self) -> None:
        """Verify we have at least 6 roleplay escape templates."""
        templates = load_jailbreak_templates()
        roleplay = [t for t in templates if t.category == AttackCategory.ROLEPLAY_ESCAPE]
        assert len(roleplay) >= 6


class TestInjectionTemplates:
    """Tests for PROMPT_INJECTION and ENCODING_OBFUSCATION template loaders."""

    def test_load_returns_non_empty_list(self) -> None:
        templates = load_injection_templates()
        assert len(templates) > 0

    def test_contains_injection_category(self) -> None:
        templates = load_injection_templates()
        categories = {t.category for t in templates}
        assert AttackCategory.PROMPT_INJECTION in categories

    def test_contains_obfuscation_category(self) -> None:
        templates = load_injection_templates()
        categories = {t.category for t in templates}
        assert AttackCategory.ENCODING_OBFUSCATION in categories

    def test_no_jailbreak_categories_in_injection_loader(self) -> None:
        templates = load_injection_templates()
        for t in templates:
            assert t.category not in {
                AttackCategory.DIRECT_JAILBREAK,
                AttackCategory.ROLEPLAY_ESCAPE,
            }

    def test_injection_template_ids_are_unique(self) -> None:
        templates = load_injection_templates()
        ids = [t.id for t in templates]
        assert len(ids) == len(set(ids))

    @pytest.mark.parametrize(
        "category",
        [AttackCategory.PROMPT_INJECTION, AttackCategory.ENCODING_OBFUSCATION],
    )
    def test_each_category_has_templates(self, category: AttackCategory) -> None:
        templates = load_injection_templates()
        found = [t for t in templates if t.category == category]
        assert len(found) >= 1, f"No templates found for category {category}"


class TestExtractionTemplates:
    """Tests for SYSTEM_EXTRACTION and TRAINING_DATA_EXTRACTION loaders."""

    def test_load_returns_non_empty_list(self) -> None:
        templates = load_extraction_templates()
        assert len(templates) > 0

    def test_contains_system_extraction(self) -> None:
        templates = load_extraction_templates()
        categories = {t.category for t in templates}
        assert AttackCategory.SYSTEM_EXTRACTION in categories

    def test_contains_training_data_extraction(self) -> None:
        templates = load_extraction_templates()
        categories = {t.category for t in templates}
        assert AttackCategory.TRAINING_DATA_EXTRACTION in categories

    def test_extraction_template_ids_are_unique(self) -> None:
        templates = load_extraction_templates()
        ids = [t.id for t in templates]
        assert len(ids) == len(set(ids))

    def test_high_severity_templates_exist(self) -> None:
        """PII extraction and API key probes should be severity 5."""
        templates = load_extraction_templates()
        high_severity = [t for t in templates if t.severity == 5]
        assert len(high_severity) >= 1


class TestAllTemplatesCombined:
    """Integration test: all loaders together cover all 6 categories."""

    def test_all_six_categories_covered(self) -> None:
        all_templates = (
            load_jailbreak_templates()
            + load_injection_templates()
            + load_extraction_templates()
        )
        covered = {t.category for t in all_templates}
        assert covered == set(AttackCategory)

    def test_total_template_count_above_minimum(self) -> None:
        """We need at least 30 templates across all categories."""
        all_templates = (
            load_jailbreak_templates()
            + load_injection_templates()
            + load_extraction_templates()
        )
        assert len(all_templates) >= 30

    def test_no_duplicate_ids_across_loaders(self) -> None:
        all_templates = (
            load_jailbreak_templates()
            + load_injection_templates()
            + load_extraction_templates()
        )
        ids = [t.id for t in all_templates]
        assert len(ids) == len(set(ids)), "Duplicate IDs found across loaders."
