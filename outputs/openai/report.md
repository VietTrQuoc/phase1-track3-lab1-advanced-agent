# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_150.json
- Mode: openai (gpt-4.1-nano)
- Records: 300 (150 ReAct + 150 Reflexion)
- Agents: react, reflexion
- Date: 2026-04-23

## Summary
| Metric | ReAct | Reflexion | Delta (Reflexion − ReAct) |
|---|---:|---:|---:|
| Exact Match (EM) | 0.7067 | 0.9000 | **+0.1933** |
| Avg attempts | 1.00 | 1.3933 | +0.3933 |
| Avg token estimate | 1755.73 | 4868.12 | +3112.39 |
| Avg latency (ms) | 1927.80 | 5169.41 | +3241.61 |

Reflexion achieves **90% EM** compared to **70.67% EM** for ReAct — a **+19.33 percentage point** improvement on 150 HotpotQA multi-hop questions using GPT-4.1-nano.

## Failure Modes

### By Agent
```json
{
  "react": {
    "none": 106,
    "wrong_final_answer": 43,
    "entity_drift": 1
  },
  "reflexion": {
    "none": 135,
    "reflection_overfit": 15
  },
  "combined": {
    "none": 241,
    "wrong_final_answer": 43,
    "reflection_overfit": 15,
    "entity_drift": 1
  }
}
```

### Analysis
| Failure Mode | ReAct | Reflexion | Description |
|---|---:|---:|---|
| none (correct) | 106 | 135 | No failure |
| wrong_final_answer | 43 | 0 | Agent answered but EM mismatch |
| reflection_overfit | 0 | 15 | Reflexion loop converged on a wrong answer |
| entity_drift | 1 | 0 | Second-hop entity substituted incorrectly |

**ReAct** fails most often via `wrong_final_answer` (43/150 = 28.7%) — the agent reaches a plausible-sounding but incorrect answer in a single pass with no chance to self-correct.

**Reflexion** nearly eliminates `wrong_final_answer` by reflecting on failures and retrying, but introduces `reflection_overfit` (15/150 = 10%) — cases where the evaluator incorrectly passes a wrong answer, causing early termination, or the reflector reinforces an incorrect strategy across retries.

## Extensions Implemented
- `structured_evaluator` — LLM evaluator returns structured JSON with `score`, `reason`, `missing_evidence`, `spurious_claims`
- `reflection_memory` — accumulated lessons from prior failed attempts are injected into subsequent actor prompts
- `benchmark_report_json` — full structured report saved as `report.json` alongside `report.md`
- `adaptive_max_attempts` — difficulty-aware attempt budget: easy→2, medium→3, hard→4
- `memory_compression` — reflection history is summarized when it exceeds a configurable limit to avoid context bloat
- `plan_then_execute` — actor first generates a numbered decomposition plan before answering, reducing entity drift

## Discussion

Reflexion delivers a clear +19.33 pp improvement over ReAct on multi-hop HotpotQA questions, rising from 70.67% to 90% exact match. The primary mechanism is the reflection memory loop: each failed attempt contributes a structured lesson (failure reason, corrective strategy) that is prepended to the next actor prompt. This allows the model to avoid repeating the same multi-hop reasoning error across retries.

The `plan_then_execute` extension further reduces entity drift by having the actor enumerate intermediate reasoning steps before committing to an answer. Combined with adaptive attempt budgets, difficult questions receive up to four attempts while easy ones terminate early, managing cost and latency.

The main residual failure mode for Reflexion is `reflection_overfit` (10%): when the structured evaluator assigns a passing score to a near-miss answer, the loop terminates prematurely. Improving evaluator strictness — for example requiring evidence citations — would likely recover most of these cases.

The cost tradeoff is significant: Reflexion uses ~2.77× more tokens (4868 vs 1756) and ~2.68× more latency (5169 ms vs 1928 ms) per question, since each retry invokes actor, evaluator, and reflector model calls. Memory compression mitigates token growth across long reflection chains by summarizing older entries, keeping per-attempt context manageable.
