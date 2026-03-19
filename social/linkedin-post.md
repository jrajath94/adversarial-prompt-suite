# LinkedIn Post

I just open-sourced adversarial-prompt-suite — here's why attack surface coverage matters more than attack count.

Every AI safety team I've spoken to runs ad-hoc red-teaming scripts. When I ask "what percentage of your attack surface have you covered?", the honest answer is usually "we don't know." That's the problem I built this to solve. Running 500 jailbreak prompts doesn't tell you whether your model is vulnerable to prompt injection or training data extraction — it just tells you one dimension of a six-dimensional problem.

adversarial-prompt-suite structures adversarial attacks into six categories (direct jailbreak, roleplay escape, prompt injection, system extraction, training data extraction, encoding obfuscation) and computes coverage metrics across all six. The classifier uses a two-layer approach: fast regex heuristics handle 80% of cases in under 1ms; a configurable LLM judge escalates only BORDERLINE cases. The result is a structured coverage report that tells you not just "how many attacks succeeded" but "which attack categories you haven't tested" — which is what actually matters for a safeguards team.

The framework runs fully offline with a mock LLM client, integrates with any OpenAI-compatible API endpoint, and emits both JSON reports (for CI pipelines) and Markdown reports (for team review). Benchmark results on my machine: 47,000+ evals/sec with the mock client; real-API throughput is rate-limited by the model endpoint, not the framework.

This is the measurement layer that safety evaluation has been missing. I'm planning to add template novelty scoring next (so you can detect when you're adding paraphrases rather than genuinely new attack vectors) and per-model comparison reports for regression tracking across model versions. If you're working on LLM safety evaluation and want to contribute or just kick the tires, it's all on GitHub.

→ GitHub: github.com/jrajath94/adversarial-prompt-suite

#AISafety #LLMSafety #RedTeaming #MachineLearning #SoftwareEngineering #OpenSource
