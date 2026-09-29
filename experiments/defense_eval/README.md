# Defense evaluation: measured results

## What was measured

Three defenses were compared against a no-defense baseline, per attack
category, on a fixed target model. The question: how much does each defense
reduce the attack success rate?

- **Target model:** qwen/qwen3-8b via OpenRouter, temperature 0.
- **Judge:** the suite's two-layer classifier (regex heuristics; borderline
  cases go to google/gemini-2.5-flash as LLM judge).
- **Defenses:**
  - `sysprompt`: a fixed hardened system prompt prepended to every request
    (see `src/adversarial_prompt_suite/defenses.py`, `SYSTEM_PROMPT_V1`).
  - `outfilter`: a regex post-filter on the raw response. If compliance
    patterns match, the response is replaced with a fixed refusal. This is
    a simple pattern filter, not a learned model.
  - `verifygate`: a second LLM call (google/gemini-2.5-flash) checks the
    response for attack compliance and returns a JSON verdict. If the
    verifier says UNSAFE, the response is replaced with a fixed refusal.
    If the verifier call fails or returns unparseable output, the response
    passes through (fail-open), and the failure is logged.
- **Sample size:** every template x 5 replicates per condition, per
  category. 10 benign prompts x 5 replicates per condition measure refusal
  inflation.
- **Metric:** attack success rate (UNSAFE verdicts / evaluated), Wilson 95%
  CI. Borderline verdicts are reported separately, never folded in.

## Results (per category)

Attack success rate by defense. n = evaluated rows per cell (error rows
excluded). Delta is percentage points vs baseline.

| Category | Baseline | Sysprompt | Outfilter | Verifygate |
|---|---|---|---|---|
| DIRECT_JAILBREAK | 0/40, 0.0% | 0/40, 0.0% (+0.0) | 0/40, 0.0% (+0.0) | 0/40, 0.0% (+0.0) |
| ROLEPLAY_ESCAPE | 8/30, 26.7% | 0/30, 0.0% (-26.7) | 8/30, 26.7% (+0.0) | 0/30, 0.0% (-26.7) |
| PROMPT_INJECTION | 0/40, 0.0% | 0/40, 0.0% (+0.0) | 0/40, 0.0% (+0.0) | 0/40, 0.0% (+0.0) |
| SYSTEM_EXTRACTION | 5/39, 12.8% | 3/40, 7.5% (-5.3) | 6/40, 15.0% (+2.2) | 1/40, 2.5% (-10.3) |
| TRAINING_DATA_EXTRACTION | 4/40, 10.0% | 0/40, 0.0% (-10.0) | 5/40, 12.5% (+2.5) | 3/40, 7.5% (-2.5) |
| ENCODING_OBFUSCATION | 0/35, 0.0% | 3/35, 8.6% (+8.6) | 0/35, 0.0% (+0.0) | 0/35, 0.0% (+0.0) |

Benign refusal: 0/50 baseline, 0/50 sysprompt, 0/50 outfilter, 0/20
verifygate (partial: benign phase interrupted by a hung network call;
attack phase 225/225 complete). No defense inflated benign refusals in
the measured samples.

## What the numbers say

- The verification gate is the strongest defense tested. It eliminated
  roleplay escapes entirely (26.7% to 0.0%) and cut system-prompt
  extraction by four-fifths (12.8% to 2.5%). It costs one extra LLM call
  per response.
- The hardened system prompt helped where the baseline was weakest:
  roleplay escapes and training-data extraction both fell to zero. But it
  backfired on encoding obfuscation, where success rose from 0.0% to 8.6%.
  A plausible read: the extra system-prompt text gives encoded attacks
  more material to hide in. Hardening is not uniformly helpful.
- The regex output filter did nothing useful. It left roleplay escapes
  untouched at 26.7% and was slightly worse than baseline on extraction
  categories. Pattern matching is not a defense here.
- Three categories sit at a floor: the target refuses direct jailbreaks,
  prompt injections, and (at baseline) encoding attacks without any
  defense. Zeros there are the model, not the defenses.

## Limits

- One target model (qwen/qwen3-8b), one judge, five replicates. Numbers
  are estimates with wide confidence intervals at this sample size.
- The verifygate fails open on verifier errors (logged, rare: 3 transient
  JSON parse failures in 225 attacks). A fail-closed gate would block more
  but risks refusing benign traffic.
- Only the defenses described above were tested. This is a measurement
  study, not a production red-team report.

## Reproduce

```bash
# Layer 2 (baseline x all categories)
python experiments/defense_eval/run.py --categories all --conditions baseline --replicates 5
# Layer 3 (all defenses x all categories)
python experiments/defense_eval/run.py --categories all --conditions sysprompt,outfilter,verifygate --replicates 5
```

Raw rows, manifests, and per-run stats live under
`experiments/defense_eval/results/<run-id>/`. Layer 2 baseline:
`20260929T045527Z-2085bb77`. Layer 3 sysprompt/outfilter:
`20260929T060525Z-e0221c2c`. Verifygate (attack complete, benign partial):
`20260929T092340Z-4974b648`.
