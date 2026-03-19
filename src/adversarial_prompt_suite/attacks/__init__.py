"""Attack template modules for adversarial-prompt-suite.

Each sub-module provides a loader function that returns a list of
AttackTemplate objects for its category. Templates use {TARGET} and
{PAYLOAD} placeholders substituted at evaluation time.
"""

from adversarial_prompt_suite.attacks.jailbreak import load_jailbreak_templates
from adversarial_prompt_suite.attacks.injection import load_injection_templates
from adversarial_prompt_suite.attacks.extraction import load_extraction_templates

__all__ = [
    "load_jailbreak_templates",
    "load_injection_templates",
    "load_extraction_templates",
]
