# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_mini.json
- Mode: mock
- Records: 16
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.5 | 1.0 | 0.5 |
| Avg attempts | 1 | 1.5 | 0.5 |
| Avg token estimate | 385 | 790 | 405 |
| Avg latency (ms) | 200 | 455 | 255 |

## Failure modes
```json
{
  "react": {
    "none": 4,
    "incomplete_multi_hop": 1,
    "wrong_final_answer": 1,
    "entity_drift": 2
  },
  "reflexion": {
    "none": 8
  }
}
```

## Extensions implemented
- structured_evaluator
- reflection_memory
- benchmark_report_json
- adaptive_max_attempts
- memory_compression
- plan_then_execute
- mock_mode_for_autograding

## Discussion
Reflexion improves hard multi-hop cases by reusing reflection memory and a plan-then-execute flow. Adaptive attempts raise effort on hard examples while limiting cost on easy ones. Memory compression keeps retries concise and reduces repeated context bloat. The tradeoff is higher latency and token usage because evaluator and reflector are also model calls. Residual failures mostly occur when the evaluator under-specifies missing evidence or when early plans anchor to a wrong entity.
