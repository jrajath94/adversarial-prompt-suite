# X (Twitter) Post Thread

## Tweet 1 (Hook)
Most red-teaming teams test their LLMs with ad-hoc scripts.

Ask them "what % of your attack surface have you covered?" and you get silence.

So I built a framework that answers that question.

Code: github.com/jrajath94/adversarial-prompt-suite
🧵

---

## Tweet 2 (The Problem)
Manual red-teaming has two failure modes:

1. You run the same jailbreak prompts over and over (low diversity)
2. You only test 1-2 attack categories and call it done (low coverage)

Neither tells you where your blind spots are.

---

## Tweet 3 (The Approach)
adversarial-prompt-suite structures attacks into 6 categories:

- DIRECT_JAILBREAK (instruction overrides)
- ROLEPLAY_ESCAPE (character bypasses)
- PROMPT_INJECTION (data-field attacks)
- SYSTEM_EXTRACTION (prompt leakage)
- TRAINING_DATA_EXTRACTION (memorization probes)
- ENCODING_OBFUSCATION (token-filter bypasses)

Then measures coverage across all 6.

---

## Tweet 4 (Key Technical Insight)
The classifier uses a 2-layer approach:

Layer 1: Regex heuristics (sub-ms, zero cost) → handles 80% of cases
Layer 2: LLM-judge (gpt-4o-mini) → only for BORDERLINE cases

This keeps evaluation cheap while maintaining accuracy.
False negatives are more costly than false positives in safety evaluation.

---

## Tweet 5 (Benchmarks)
Mock client benchmark:

batch=100, concurrency=50: 47,000+ evals/sec (framework overhead)
Real API bottleneck: ~500ms per batch of 100 at concurrency=20

Category coverage with full template set: 100% (all 6 categories)
Built-in templates: 37 across 6 categories

---

## Tweet 6 (CTA)
Star it if useful. What should I build next?

github.com/jrajath94/adversarial-prompt-suite

#AISafety #LLM #RedTeaming #MachineLearning #OpenSource #BuildInPublic
