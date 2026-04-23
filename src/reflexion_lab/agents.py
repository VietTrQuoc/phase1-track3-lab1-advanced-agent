from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

from .llm_runtime import (
    actor_answer as llm_actor_answer,
    evaluator as llm_evaluator,
    get_openai_model,
    get_ollama_model,
    plan_then_execute_answer,
    reflector as llm_reflector,
)
from .mock_runtime import (
    FAILURE_MODE_BY_QID,
    actor_answer as mock_actor_answer,
    evaluator as mock_evaluator,
    reflector as mock_reflector,
)
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord


def _compress_memory(reflection_memory: list[str], limit: int = 3) -> list[str]:
    if len(reflection_memory) <= limit:
        return reflection_memory

    older = reflection_memory[:-2]
    recent = reflection_memory[-2:]
    compressed = "Compressed lessons: " + " | ".join(item.split("|", 1)[0].strip() for item in older)
    return [compressed, *recent]


def _adaptive_attempts(difficulty: str, base_attempts: int) -> int:
    mapping = {"easy": 2, "medium": 3, "hard": 4}
    return max(base_attempts, mapping.get(difficulty, base_attempts))


@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    runtime_mode: Literal["mock", "ollama", "openai"] = "mock"
    ollama_model: str = ""
    use_adaptive_attempts: bool = False
    use_memory_compression: bool = False
    use_plan_then_execute: bool = False

    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0

        max_attempts = self.max_attempts
        if self.use_adaptive_attempts and self.agent_type == "reflexion":
            max_attempts = _adaptive_attempts(example.difficulty, self.max_attempts)

        model = self.ollama_model
        if not model:
            model = get_openai_model() if self.runtime_mode == "openai" else get_ollama_model()

        for attempt_id in range(1, max_attempts + 1):
            attempt_tokens = 0
            attempt_latency = 0
            plan = ""

            if self.runtime_mode in {"ollama", "openai"}:
                if self.use_plan_then_execute:
                    actor_result = plan_then_execute_answer(
                        example,
                        attempt_id,
                        reflection_memory,
                        model,
                        runtime_mode=self.runtime_mode,
                    )
                else:
                    actor_result = llm_actor_answer(
                        example,
                        attempt_id,
                        reflection_memory,
                        model,
                        runtime_mode=self.runtime_mode,
                    )
                answer = actor_result.answer
                plan = actor_result.plan
                attempt_tokens += actor_result.token_count
                attempt_latency += actor_result.latency_ms

                judge, eval_tokens, eval_latency = llm_evaluator(
                    example,
                    answer,
                    model,
                    runtime_mode=self.runtime_mode,
                )
                attempt_tokens += eval_tokens
                attempt_latency += eval_latency
            else:
                answer = mock_actor_answer(example, attempt_id, self.agent_type, reflection_memory)
                judge = mock_evaluator(example, answer)
                attempt_tokens = 320 + (attempt_id * 65) + (120 if self.agent_type == "reflexion" else 0)
                attempt_latency = 160 + (attempt_id * 40) + (90 if self.agent_type == "reflexion" else 0)

            trace = AttemptTrace(
                attempt_id=attempt_id,
                plan=plan,
                answer=answer,
                score=judge.score,
                reason=judge.reason,
                reflection_memory_snapshot=list(reflection_memory),
                token_estimate=attempt_tokens,
                latency_ms=attempt_latency,
            )
            final_answer = answer
            final_score = judge.score

            if judge.score == 1:
                traces.append(trace)
                break

            if self.agent_type == "reflexion" and attempt_id < max_attempts:
                if self.runtime_mode in {"ollama", "openai"}:
                    reflection, ref_tokens, ref_latency = llm_reflector(
                        example,
                        attempt_id,
                        judge,
                        answer,
                        model,
                        runtime_mode=self.runtime_mode,
                    )
                    trace.token_estimate += ref_tokens
                    trace.latency_ms += ref_latency
                else:
                    reflection = mock_reflector(example, attempt_id, judge)

                trace.reflection = reflection
                reflections.append(reflection)
                reflection_memory.append(
                    f"lesson: {reflection.lesson} | strategy: {reflection.strategy} | why_failed: {reflection.failure_reason}"
                )
                if self.use_memory_compression:
                    reflection_memory = _compress_memory(reflection_memory)

            traces.append(trace)

        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        failure_mode = "none" if final_score == 1 else FAILURE_MODE_BY_QID.get(example.qid, "wrong_final_answer")

        if self.agent_type == "reflexion" and len(reflections) > 1 and final_score == 0:
            failure_mode = "reflection_overfit"

        return RunRecord(
            qid=example.qid,
            question=example.question,
            gold_answer=example.gold_answer,
            agent_type=self.agent_type,
            predicted_answer=final_answer,
            is_correct=bool(final_score),
            attempts=len(traces),
            token_estimate=total_tokens,
            latency_ms=total_latency,
            failure_mode=failure_mode,
            reflections=reflections,
            traces=traces,
        )


class ReActAgent(BaseAgent):
    def __init__(self, runtime_mode: Literal["mock", "ollama", "openai"] = "mock", ollama_model: str = "") -> None:
        super().__init__(agent_type="react", max_attempts=1, runtime_mode=runtime_mode, ollama_model=ollama_model)


class ReflexionAgent(BaseAgent):
    def __init__(
        self,
        max_attempts: int = 3,
        runtime_mode: Literal["mock", "ollama", "openai"] = "mock",
        ollama_model: str = "",
        adaptive_attempts: bool = True,
        memory_compression: bool = True,
        plan_then_execute: bool = True,
    ) -> None:
        super().__init__(
            agent_type="reflexion",
            max_attempts=max_attempts,
            runtime_mode=runtime_mode,
            ollama_model=ollama_model,
            use_adaptive_attempts=adaptive_attempts,
            use_memory_compression=memory_compression,
            use_plan_then_execute=plan_then_execute,
        )
