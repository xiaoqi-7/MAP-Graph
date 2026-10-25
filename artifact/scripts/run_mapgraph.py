import argparse
import json
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.autogen_workflow import AutoGenWorkflow, WorkflowConfig, save_workflow_outputs
from src.map_graph import MAPGraphMemoryBackend


def load_tasks(tasks_path: str, task_id: str | None = None) -> list[dict]:
    path = Path(tasks_path)
    files = [path] if path.is_file() else sorted(path.glob("*.json"))
    tasks = []
    for file in files:
        with file.open("r", encoding="utf-8") as f:
            task = json.load(f)
        if task_id is None or task.get("task_id") == task_id:
            tasks.append(task)
    if not tasks:
        raise ValueError(f"No tasks found in {tasks_path!r}" + (f" for task_id={task_id!r}" if task_id else ""))
    return tasks


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"


def decision_summary(decisions: list[dict]) -> str:
    if not decisions:
        return "none"
    counts: dict[str, int] = {}
    for decision in decisions:
        label = str(decision.get("decision") or "UNKNOWN")
        counts[label] = counts.get(label, 0) + 1
    return ",".join(f"{label}:{counts[label]}" for label in sorted(counts))


def log_progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def save_map_graph_outputs(result: dict, memory_backend: MAPGraphMemoryBackend, output_dir: str | Path) -> None:
    result["memory_log"]["map_graph_summary"] = memory_backend.graph_summary()
    save_workflow_outputs(result, output_dir)

    output_root = Path(output_dir)
    run_id = result.get("trace", {}).get("run_id")
    graph_dir = output_root / "map_graphs" / result["method"]
    if run_id:
        graph_dir = graph_dir / run_id
    graph_dir.mkdir(parents=True, exist_ok=True)
    with (graph_dir / f"{result['task_id']}.json").open("w", encoding="utf-8") as f:
        json.dump(memory_backend.export_graph(), f, ensure_ascii=False, indent=2)


def output_paths(output_dir: str | Path, method: str, run_id: str | None, task_id: str) -> dict[str, Path]:
    output_root = Path(output_dir)

    def method_dir(kind: str) -> Path:
        root = output_root / kind / method
        return root / run_id if run_id else root

    graph_dir = output_root / "map_graphs" / method
    if run_id:
        graph_dir = graph_dir / run_id

    return {
        "trace": method_dir("traces") / f"{task_id}.json",
        "memory_log": method_dir("memory_logs") / f"{task_id}.json",
        "decision_log": method_dir("decisions") / f"{task_id}.json",
        "graph": graph_dir / f"{task_id}.json",
    }


def task_outputs_exist(output_dir: str | Path, method: str, run_id: str | None, task_id: str) -> bool:
    return all(path.exists() for path in output_paths(output_dir, method, run_id, task_id).values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="data/final_tasks")
    parser.add_argument("--output-dir", default="outputs_final_paper")
    parser.add_argument("--run-id")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--run-prefix", default="run")
    parser.add_argument("--task-id")
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument("--model-max-tokens", type=int, default=2048)
    parser.add_argument("--model-timeout", type=float, default=120)
    parser.add_argument("--model-max-retries", type=int, default=5)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    method = "MAPGraph"
    tasks = load_tasks(args.tasks, task_id=args.task_id)
    run_ids = [args.run_id] if args.run_id else [f"{args.run_prefix}{i}" for i in range(1, args.runs + 1)]
    total_tasks = len(tasks)
    total_work = len(run_ids) * total_tasks
    finished_work = 0
    experiment_start = time.perf_counter()
    all_completed = {}

    log_progress(
        "[mapgraph:start] "
        f"method={method} runs={len(run_ids)} tasks_per_run={total_tasks} total_tasks={total_work}"
    )

    for run_index, run_id in enumerate(run_ids, start=1):
        completed = []
        memory_backend = MAPGraphMemoryBackend()
        run_start = time.perf_counter()
        log_progress(
            "[mapgraph:run_start] "
            f"method={method} run={run_id} run_index={run_index}/{len(run_ids)} tasks={total_tasks}"
        )
        for task_index, task in enumerate(tasks, start=1):
            task_id = str(task.get("task_id", "unknown_task"))
            if args.skip_existing and task_outputs_exist(args.output_dir, method, run_id, task_id):
                finished_work += 1
                completed.append(
                    {
                        "task_id": task_id,
                        "domain": task.get("domain"),
                        "method": method,
                        "decisions": "skipped_existing",
                    }
                )
                run_elapsed = time.perf_counter() - run_start
                total_elapsed = time.perf_counter() - experiment_start
                avg_total = total_elapsed / max(1, finished_work)
                eta_total = avg_total * (total_work - finished_work)
                eta_run = (run_elapsed / task_index) * (total_tasks - task_index) if task_index else 0
                log_progress(
                    "[mapgraph:skip] "
                    f"method={method} run={run_id} run_index={run_index}/{len(run_ids)} "
                    f"task={task_index}/{total_tasks} total={finished_work}/{total_work} "
                    f"task_id={task_id} domain={task.get('domain')} "
                    f"run_elapsed={format_duration(run_elapsed)} run_eta={format_duration(eta_run)} "
                    f"total_elapsed={format_duration(total_elapsed)} total_eta={format_duration(eta_total)}"
                )
                continue
            task_start = time.perf_counter()
            workflow = AutoGenWorkflow(
                memory_backend=memory_backend,
                config=WorkflowConfig(
                    max_rounds=args.max_rounds,
                    method_name=method,
                    model_max_tokens=args.model_max_tokens,
                    model_timeout=args.model_timeout,
                    model_max_retries=args.model_max_retries,
                ),
            )
            result = workflow.run_task_sync(task)
            result["trace"]["run_id"] = run_id
            result["memory_log"]["run_id"] = run_id
            result["decision_log"]["run_id"] = run_id
            save_map_graph_outputs(result, memory_backend, args.output_dir)

            finished_work += 1
            task_elapsed = time.perf_counter() - task_start
            run_elapsed = time.perf_counter() - run_start
            total_elapsed = time.perf_counter() - experiment_start
            avg_total = total_elapsed / finished_work
            eta_total = avg_total * (total_work - finished_work)
            eta_run = (run_elapsed / task_index) * (total_tasks - task_index) if task_index else 0
            decisions = result["decision_log"]["decisions"]
            completed.append(
                {
                    "task_id": result["task_id"],
                    "domain": result["domain"],
                    "method": result["method"],
                    "decisions": len(decisions),
                }
            )
            log_progress(
                "[mapgraph:task] "
                f"method={method} run={run_id} run_index={run_index}/{len(run_ids)} "
                f"task={task_index}/{total_tasks} total={finished_work}/{total_work} "
                f"task_id={result['task_id']} domain={result['domain']} "
                f"decisions={decision_summary(decisions)} "
                f"task_time={format_duration(task_elapsed)} "
                f"run_elapsed={format_duration(run_elapsed)} run_eta={format_duration(eta_run)} "
                f"total_elapsed={format_duration(total_elapsed)} total_eta={format_duration(eta_total)}"
            )
        all_completed[run_id] = completed
        log_progress(
            "[mapgraph:run_done] "
            f"method={method} run={run_id} completed={len(completed)}/{total_tasks} "
            f"run_elapsed={format_duration(time.perf_counter() - run_start)}"
        )

    log_progress(
        "[mapgraph:done] "
        f"method={method} runs={len(run_ids)} completed={finished_work}/{total_work} "
        f"elapsed={format_duration(time.perf_counter() - experiment_start)}"
    )
    print(json.dumps({"method": method, "run_ids": run_ids, "completed": all_completed}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
