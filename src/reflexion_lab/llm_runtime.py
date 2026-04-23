from __future__ import annotations

import json
import time
from dataclasses import dataclass
from os import getenv
from typing import Any, Literal

from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, PLAN_SYSTEM, REFLECTOR_SYSTEM
from .schemas import JudgeResult, QAExample, ReflectionEntry
from .utils import normalize_answer

try:
    import ollama
except Exception:  # pragma: no cover - optional dependency during mock-only runs
    ollama = None

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency during ollama/mock runs
    OpenAI = None


@dataclass
class RuntimeAnswer:
    answer: str
    token_count: int
    latency_ms: int
    plan: str = ""


def _format_context(example: QAExample) -> str:
    lines: list[str] = []
    for idx, chunk in enumerate(example.context, start=1):
        lines.append(f"[{idx}] {chunk.title}: {chunk.text}")
    return "\n".join(lines)


def _safe_json_loads(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json\n", "", 1)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _estimate_tokens_from_text(text: str) -> int:
    # Fallback token estimate when provider usage stats are missing.
    return max(1, len(text) // 4)


def _usage_tokens(resp: dict[str, Any], content: str) -> int:
    prompt_count = int(resp.get("prompt_eval_count") or 0)
    eval_count = int(resp.get("eval_count") or 0)
    total = prompt_count + eval_count
    if total > 0:
        return total
    return _estimate_tokens_from_text(content)


def _usage_tokens_openai(resp: Any, content: str) -> int:
    usage = getattr(resp, "usage", None)
    if usage is not None:
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total = prompt_tokens + completion_tokens
        if total > 0:
            return total
    return _estimate_tokens_from_text(content)


def _ollama_chat(model: str, system: str, user_prompt: str, temperature: float = 0.0) -> tuple[str, int, int]:
    if ollama is None:
        raise RuntimeError("Ollama package is not installed. Run: pip install ollama")

    start = time.perf_counter()
    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": temperature},
    )
    latency_ms = int((time.perf_counter() - start) * 1000)
    message = resp.get("message", {})
    content = str(message.get("content", "")).strip()
    tokens = _usage_tokens(resp, content)
    return content, tokens, latency_ms


def _openai_chat(model: str, system: str, user_prompt: str, temperature: float = 0.0) -> tuple[str, int, int]:
    if OpenAI is None:
        raise RuntimeError("OpenAI package is not installed. Run: pip install openai")

    api_key = getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Please set it in your environment or .env file.")

    base_url = getenv("OPENAI_BASE_URL", "").strip() or None
    client = OpenAI(api_key=api_key, base_url=base_url)

    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
    )
    latency_ms = int((time.perf_counter() - start) * 1000)
    content = (resp.choices[0].message.content or "").strip()
    tokens = _usage_tokens_openai(resp, content)
    return content, tokens, latency_ms


def _chat(
    runtime_mode: Literal["ollama", "openai"],
    model: str,
    system: str,
    user_prompt: str,
    temperature: float = 0.0,
) -> tuple[str, int, int]:
    if runtime_mode == "openai":
        return _openai_chat(model=model, system=system, user_prompt=user_prompt, temperature=temperature)
    return _ollama_chat(model=model, system=system, user_prompt=user_prompt, temperature=temperature)


def _build_actor_prompt(example: QAExample, attempt_id: int, reflection_memory: list[str], plan: str) -> str:
    reflection_blob = "\n".join(f"- {item}" for item in reflection_memory) if reflection_memory else "- none"
    return (
        f"Question:\n{example.question}\n\n"
        f"Context:\n{_format_context(example)}\n\n"
        f"Attempt ID: {attempt_id}\n"
        f"Reflection memory:\n{reflection_blob}\n\n"
        f"Plan:\n{plan or 'No plan provided.'}\n\n"
        "Return only the final answer string."
    )


def plan_then_execute_answer(
    example: QAExample,
    attempt_id: int,
    reflection_memory: list[str],
    model: str,
    runtime_mode: Literal["ollama", "openai"] = "ollama",
) -> RuntimeAnswer:
    planner_prompt = (
        f"Question:\n{example.question}\n\n"
        f"Context:\n{_format_context(example)}\n\n"
        f"Attempt ID: {attempt_id}\n"
        f"Known reflections:\n{chr(10).join(f'- {m}' for m in reflection_memory) if reflection_memory else '- none'}"
    )
    plan_text, plan_tokens, plan_latency = _chat(
        runtime_mode=runtime_mode,
        model=model,
        system=PLAN_SYSTEM,
        user_prompt=planner_prompt,
        temperature=0.0,
    )

    actor_prompt = _build_actor_prompt(example=example, attempt_id=attempt_id, reflection_memory=reflection_memory, plan=plan_text)
    answer_text, answer_tokens, answer_latency = _chat(
        runtime_mode=runtime_mode,
        model=model,
        system=ACTOR_SYSTEM,
        user_prompt=actor_prompt,
        temperature=0.0,
    )

    return RuntimeAnswer(
        answer=answer_text,
        plan=plan_text,
        token_count=plan_tokens + answer_tokens,
        latency_ms=plan_latency + answer_latency,
    )


def actor_answer(
    example: QAExample,
    attempt_id: int,
    reflection_memory: list[str],
    model: str,
    runtime_mode: Literal["ollama", "openai"] = "ollama",
) -> RuntimeAnswer:
    user_prompt = _build_actor_prompt(example=example, attempt_id=attempt_id, reflection_memory=reflection_memory, plan="")
    answer_text, tokens, latency = _chat(
        runtime_mode=runtime_mode,
        model=model,
        system=ACTOR_SYSTEM,
        user_prompt=user_prompt,
        temperature=0.0,
    )
    return RuntimeAnswer(answer=answer_text, token_count=tokens, latency_ms=latency)


def evaluator(
    example: QAExample,
    answer: str,
    model: str,
    runtime_mode: Literal["ollama", "openai"] = "ollama",
) -> tuple[JudgeResult, int, int]:
    user_prompt = (
        f"Question:\n{example.question}\n\n"
        f"Gold answer:\n{example.gold_answer}\n\n"
        f"Model answer:\n{answer}\n"
    )

    content, tokens, latency = _chat(
        runtime_mode=runtime_mode,
        model=model,
        system=EVALUATOR_SYSTEM,
        user_prompt=user_prompt,
        temperature=0.0,
    )
    data = _safe_json_loads(content)

    if not data:
        score = int(normalize_answer(example.gold_answer) == normalize_answer(answer))
        reason = "Evaluator returned invalid JSON; used normalization fallback."
        judge = JudgeResult(score=score, reason=reason)
        return judge, tokens, latency

    try:
        judge = JudgeResult.model_validate(data)
    except Exception:
        score = int(normalize_answer(example.gold_answer) == normalize_answer(answer))
        reason = "Evaluator JSON shape invalid; used normalization fallback."
        judge = JudgeResult(score=score, reason=reason)
    return judge, tokens, latency


def reflector(
    example: QAExample,
    attempt_id: int,
    judge: JudgeResult,
    answer: str,
    model: str,
    runtime_mode: Literal["ollama", "openai"] = "ollama",
) -> tuple[ReflectionEntry, int, int]:
    user_prompt = (
        f"Question:\n{example.question}\n\n"
        f"Answer from attempt {attempt_id}:\n{answer}\n\n"
        f"Evaluator reason:\n{judge.reason}\n\n"
        f"Missing evidence:\n{json.dumps(judge.missing_evidence)}\n\n"
        f"Spurious claims:\n{json.dumps(judge.spurious_claims)}\n"
    )

    content, tokens, latency = _chat(
        runtime_mode=runtime_mode,
        model=model,
        system=REFLECTOR_SYSTEM,
        user_prompt=user_prompt,
        temperature=0.1,
    )
    data = _safe_json_loads(content)

    if not data:
        reflection = ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=judge.reason,
            lesson="Validate each hop before finalizing.",
            strategy="Trace the chain explicitly and verify final entity against context.",
        )
        return reflection, tokens, latency

    data["attempt_id"] = attempt_id
    try:
        reflection = ReflectionEntry.model_validate(data)
    except Exception:
        reflection = ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=judge.reason,
            lesson="Fix incomplete or drifting second-hop reasoning.",
            strategy="Re-check second-hop evidence and output one grounded final entity.",
        )
    return reflection, tokens, latency


def get_runtime_mode() -> str:
    return getenv("LAB_RUNTIME_MODE", "mock").strip().lower()


def get_ollama_model() -> str:
    return getenv("OLLAMA_MODEL", "gemma4:e4b")


def get_openai_model() -> str:
    return getenv("OPENAI_MODEL", "gpt-4.1-nano")
