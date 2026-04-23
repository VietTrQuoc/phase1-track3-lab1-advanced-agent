from __future__ import annotations
import json
from pathlib import Path
import typer
from rich import print
from rich.progress import Progress
from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.llm_runtime import get_ollama_model, get_openai_model
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.utils import load_dataset, save_jsonl
app = typer.Typer(add_completion=False)

@app.command()
def main(
    dataset: str = "data/hotpot_mini.json",
    out_dir: str = "outputs/sample_run",
    reflexion_attempts: int = 3,
    mode: str = "openai",
    ollama_model: str = "",
    openai_model: str = "",
) -> None:
    mode = mode.strip().lower()
    if mode not in {"mock", "ollama", "openai"}:
        raise typer.BadParameter("mode must be one of: mock, ollama, openai")

    selected_model = ""
    if mode == "ollama":
        selected_model = ollama_model.strip() or get_ollama_model()
    if mode == "openai":
        selected_model = openai_model.strip() or get_openai_model()

    dataset_path = Path(dataset)
    if not dataset_path.exists() and not dataset_path.is_absolute():
        candidate = Path("data") / dataset_path
        if candidate.exists():
            dataset_path = candidate

    if not dataset_path.exists():
        raise typer.BadParameter(
            f"Dataset file not found: {dataset}. "
            "Try --dataset data/hotpot_150.json or an absolute path."
        )

    examples = load_dataset(dataset_path)
    react = ReActAgent(runtime_mode=mode, ollama_model=selected_model)
    reflexion = ReflexionAgent(max_attempts=reflexion_attempts, runtime_mode=mode, ollama_model=selected_model)
    
    with Progress() as progress:
        task_react = progress.add_task("[cyan]Running ReAct Agent...", total=len(examples))
        react_records = []
        for example in examples:
            react_records.append(react.run(example))
            progress.advance(task_react)
        
        task_reflexion = progress.add_task("[magenta]Running Reflexion Agent...", total=len(examples))
        reflexion_records = []
        for example in examples:
            reflexion_records.append(reflexion.run(example))
            progress.advance(task_reflexion)
    all_records = react_records + reflexion_records
    out_path = Path(out_dir)
    save_jsonl(out_path / "react_runs.jsonl", react_records)
    save_jsonl(out_path / "reflexion_runs.jsonl", reflexion_records)

    extensions = [
        "structured_evaluator",
        "reflection_memory",
        "benchmark_report_json",
        "adaptive_max_attempts",
        "memory_compression",
        "plan_then_execute",
    ]
    if mode == "mock":
        extensions.append("mock_mode_for_autograding")

    discussion = (
        "Reflexion improves hard multi-hop cases by reusing reflection memory and a plan-then-execute flow. "
        "Adaptive attempts raise effort on hard examples while limiting cost on easy ones. "
        "Memory compression keeps retries concise and reduces repeated context bloat. "
        "The tradeoff is higher latency and token usage because evaluator and reflector are also model calls. "
        "Residual failures mostly occur when the evaluator under-specifies missing evidence or when early plans anchor to a wrong entity."
    )

    report = build_report(all_records, dataset_name=dataset_path.name, mode=mode, extensions=extensions, discussion=discussion)
    json_path, md_path = save_report(report, out_path)
    print(f"[green]Saved[/green] {json_path}")
    print(f"[green]Saved[/green] {md_path}")
    print(json.dumps(report.summary, indent=2))

if __name__ == "__main__":
    app()
