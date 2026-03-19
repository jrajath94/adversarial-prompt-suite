# Interview Prep: adversarial-prompt-suite

## Elevator Pitch (30 seconds)

I built a structured red-teaming framework for LLMs that maps adversarial attacks
across six distinct categories and computes attack surface coverage metrics.
The key insight is that "does this prompt fail?" is the wrong question —
"what percentage of the known attack surface have you covered, and where are the blind spots?"
is what actually matters for a safeguards team.

## Why I Built This

### The Real Motivation

Every safeguards team I've spoken to tests their models manually or with ad-hoc scripts.
When I asked how they measure coverage, the answer was usually "we run a few hundred prompts
and check if anything breaks." That's not coverage — that's sampling. I wanted to build
something that treats safety evaluation the way software engineers treat test coverage:
structured, measurable, and auditable.

### Company-Specific Framing

| Company | Why This Matters to Them |
|---------|-------------------------|
| Anthropic | The safeguards team needs systematic coverage, not ad-hoc testing. This framework maps directly to the kind of structured evaluation that goes into Constitutional AI critique-revision loops. Coverage metrics tell you whether your safety training has blind spots. |
| OpenAI | Red-teaming at scale requires systematic methodology. This framework provides the measurement layer — you can't improve what you can't measure. |
| DeepMind | Research into adversarial robustness requires reproducible attack benchmarks. The taxonomy maps to published threat models in alignment literature. |
| NVIDIA | Inference infrastructure teams need to validate that safety layers hold across model versions and quantization schemes. This framework provides the regression test harness. |
| Google | Trust & Safety at scale means you need automated coverage reporting. This is the tool that generates that report. |
| Meta FAIR | Open safety research needs open benchmarks. This framework is the evaluation layer for published safety interventions. |
| Citadel/JS/2Sig | Adversarial evaluation of financial AI systems follows the same pattern — structured threat modelling, coverage measurement, systematic regression testing. |

## Architecture Deep-Dive

### Pipeline

```
Templates → Evaluator (asyncio) → LLM → EvaluationResult[]
                                           ↓
                              HeuristicClassifier (regex, fast)
                                           ↓ BORDERLINE only
                              LLMJudgeClassifier (gpt-4o-mini)
                                           ↓
                              compute_coverage() → CoverageReport
                                           ↓
                              ReportGenerator → JSON / Markdown
```

### Key Design Decisions

| Decision | Why | Alternative | Tradeoff |
|----------|-----|-------------|----------|
| Two-layer classifier | 80% of cases are clear-cut; LLM judge is expensive | LLM judge for all | Accuracy vs. cost |
| asyncio semaphore | Rate limit compliance without threads | Thread pool | Lower overhead, cleaner cancellation |
| Pydantic models everywhere | Schema validation at every boundary catches bugs early | TypedDict | Slightly more verbose |
| JSONL for persistence | Git-diffable, streamable, grep-able | SQLite | No transaction support |
| Six fixed categories | Maps to published threat taxonomy | Open-ended tagging | Easier to measure coverage |

### Scaling Analysis

- **Current capacity:** ~500 templates/run on a single machine with concurrency=20
- **10x strategy:** Distribute evaluation workers across multiple processes; share results via Redis queue
- **100x strategy:** Template-sharded evaluation across a Ray cluster; async result aggregation; coverage computed from partitioned results using HyperLogLog for cardinality estimation
- **Bottleneck:** LLM API rate limits (not the framework itself)
- **Cost estimate:** At 1000 templates/run with 20% LLM judge escalation: ~$0.02 at gpt-4o-mini pricing

## 10 Deep-Dive Interview Questions

### Q1: Walk me through how the classifier works end-to-end.

**A:** When an EvaluationResult arrives, `SafetyClassifier.classify()` first runs `HeuristicClassifier.classify()`. This applies two compiled regex sets: `_REFUSAL_RE` (15 patterns covering "I cannot", "I'm not able to", etc.) and `_COMPLIANCE_RE` (8 patterns covering "Sure, here is how", "Step 1:", etc.). If only refusal patterns match, we return SAFE with 0.90 confidence. If only compliance patterns match, UNSAFE at 0.85. If both match (partial compliance + caveat), or neither matches, we return BORDERLINE. BORDERLINE cases are escalated to `LLMJudgeClassifier`, which sends a structured prompt to gpt-4o-mini asking for a JSON verdict with confidence and reasoning. The entire heuristic path is sub-millisecond; the LLM judge adds ~200ms latency only for the ~20% of cases that need it.

### Q2: Why regex heuristics instead of a fine-tuned classifier?

**A:** Three reasons. First, interpretability — when a classification is wrong, I can immediately see which regex fired or didn't. With a neural classifier, debugging a false negative means examining activations. Second, zero inference cost — the heuristic runs in microseconds with no GPU or API dependency. Third, coverage: the regex patterns map directly to known safe/unsafe response structures, so I can reason about what they do and don't cover. The LLM judge handles the long tail where heuristics aren't sufficient.

### Q3: What was the hardest bug you hit?

**A:** The asyncio semaphore was being instantiated in `__init__` rather than lazily on first use. This meant that if you called `evaluate_batch` from within an already-running event loop (common in Jupyter notebooks), the semaphore was bound to the wrong loop and raised a RuntimeError. The fix was to instantiate the semaphore lazily inside `_get_semaphore()`, which creates it in the context of the current running loop. The symptom was intermittent `RuntimeError: bound to a different event loop` — took a while to connect to the semaphore lifecycle.

### Q4: How would you scale this to 100x?

**A:** Move evaluation to a Ray cluster. The template list shards trivially across workers — each worker gets a slice, evaluates it, and writes results to a shared object store. The classifier runs independently on each worker (it's stateless). Coverage computation aggregates the distributed results. The only shared state is the CoverageReport, which can be computed from a final reduce step. The LLM judge becomes a bottleneck at high concurrency — I'd rate-limit it via a token bucket and batch judge calls into larger payloads.

### Q5: What would you do differently with more time?

**A:** Three things. First, add a "novelty score" for templates — using embedding similarity to measure how far a new template is from the existing set, so you can detect when you're just adding paraphrases of existing attacks rather than genuinely new attack surfaces. Second, build a feedback loop: when the LLM judge returns BORDERLINE, queue those cases for human review, and use the human verdicts to fine-tune the heuristic patterns. Third, add per-model comparison reports — the real value is comparing safety regressions across model versions, not just a single snapshot.

### Q6: How does this compare to PromptBench or JailbreakBench?

**A:** PromptBench and JailbreakBench focus on benchmark accuracy — "does this prompt bypass the model?" with a fixed dataset. My framework focuses on coverage measurement — "what fraction of the attack surface have you exercised?" PromptBench has a larger fixed dataset but no mechanism for measuring whether you've covered the full attack space. JailbreakBench has better attack quality but no coverage analytics. The key differentiator here is the coverage report: it tells you not just "how many attacks succeeded" but "which attack categories you haven't tested."

### Q7: What are the security implications of this tool itself?

**A:** The template library contains structural attack patterns, not working exploits. The templates use `{TARGET}` and `{PAYLOAD}` placeholders that require a real payload to be a functional attack — the framework doesn't supply those. More practically: the tool requires an API key for real LLM evaluation, and those credentials should be kept in environment variables (never hardcoded). The report output includes prompt/response pairs that could contain sensitive information if run against a production model with a real system prompt — reports should be treated as confidential.

### Q8: Explain your testing strategy.

**A:** Four layers. First, unit tests for each attack loader verify template counts, category correctness, uniqueness of IDs, and schema validation. Second, classifier unit tests cover each heuristic branch with parametrized cases (safe/unsafe/borderline responses, mixed signals, error cases). Third, coverage unit tests verify metric calculations — category coverage fractions, attack success rates, template diversity — with known ground truth inputs. Fourth, integration tests run the Evaluator + SafetyClassifier together using MockLLMClient to verify the full pipeline produces correct classifications without any API calls.

### Q9: What are the failure modes?

**A:** Three main ones. First, false negatives in classification — an unsafe response that doesn't trigger compliance patterns. The LLM judge catches most of these, but novel response structures can slip through. Mitigation: monitor BORDERLINE rate; spikes indicate emerging attack patterns that need new heuristics. Second, API rate limits during batch evaluation — the semaphore prevents hammering the API, but sustained high-volume runs will still hit limits. Mitigation: exponential backoff (not yet implemented; current version captures the error in `result.error`). Third, template coverage gaps — the built-in templates cover known attack categories but can't anticipate novel attacks. Mitigation: the JSONL dataset format lets teams add new templates without code changes.

### Q10: Explain attack surface coverage from first principles.

**A:** Think of the adversarial attack space as a six-dimensional space, where each dimension is an attack category. Coverage measures how much of that space your evaluation samples. A red-team run that only tests jailbreak prompts has 1/6 category coverage, no matter how many jailbreak prompts it runs. Template diversity measures within-category coverage: if you run 100 prompts but they're all paraphrases of the same override instruction, your diversity is near zero. The goal is to maximize both: test all six categories with structurally distinct prompts. The coverage score is a proxy for "how surprised would you be by an attack in production?" — high coverage means fewer surprises.

## Complexity Analysis

- **Time (heuristic classify):** O(n * m) where n = response length, m = number of regex patterns. In practice sub-millisecond per response.
- **Time (batch evaluation):** O(templates / concurrency * latency_per_request). With concurrency=20 and 100ms avg latency: 100 templates in ~500ms.
- **Space:** O(templates + results). Each EvaluationResult is ~2KB. 1000 results = ~2MB in memory.
- **Network:** 2 API calls per template (1 evaluation, at most 1 LLM judge). Judge is called for ~20% of results.
- **Disk:** JSONL output is ~1KB per result. 1000 results = ~1MB on disk.

## Metrics (Mock Client Benchmark)

| Metric | Value | How Measured | Significance |
|--------|-------|-------------|-------------|
| Peak throughput | ~50,000 evals/sec | bench_evaluator.py (mock) | Framework overhead is minimal; bottleneck is the LLM API |
| Heuristic classify latency | <1ms | Manual timing | Zero inference cost for 80% of cases |
| Template count (built-in) | 37 | count of loader outputs | Covers all 6 categories |
| Test coverage | 88%+ | pytest-cov | Quality signal |

## Career Narrative

- **Goldman Sachs (quant research):** Safety-critical systems thinking — when a risk model fails, consequences are severe. The same mindset applies to safety classification: false negatives are costly, and you need principled measurement.
- **NVIDIA:** GPU inference work gave me a deep understanding of the infrastructure that runs these models. Understanding the deployment context informs which attack surfaces matter most.
- **JPMC (current):** Deployed production AI to 300+ users. Safety evaluation isn't abstract — it's the difference between a trustworthy system and a liability.
- **This project:** Demonstrates that I think about AI safety as a systems engineering problem, not just a policy problem. Anthropic's safeguards team wants engineers who can build the measurement layer, not just run prompts manually.

## Interview Red Flags to Avoid

- NEVER say "I built this to learn about red-teaming" — the real motivation is attack surface measurement
- NEVER claim the heuristics are perfect — acknowledge the false negative problem and how the LLM judge mitigates it
- NEVER be unable to explain why BORDERLINE exists as a third class — it's the escalation trigger
- ALWAYS connect to the company's specific safety challenges (Constitutional AI for Anthropic, RLHF eval for OpenAI)
- ALWAYS mention the coverage metric as the differentiating insight vs. existing red-team tools
- ALWAYS discuss the template novelty problem (paraphrases don't add coverage) proactively
