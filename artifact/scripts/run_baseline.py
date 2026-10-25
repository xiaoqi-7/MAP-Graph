import argparse
import json
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.autogen_workflow import (
    AutoGenWorkflow,
    WorkflowConfig,
    sanitize_runtime_task,
    save_workflow_outputs,
)
from src.memory_baselines.access_controlled_memory import AccessControlledMemoryBackend
from src.memory_baselines.flat_provenance import FlatProvenanceMemoryBackend
from src.memory_baselines.gmemory_graph import GMemoryGraphBackend
from src.memory_baselines.isolated_vector import PerAgentIsolatedMemoryBackend
from src.memory_baselines.lineage_enforcement import LineageEnforcementMemoryBackend
from src.memory_baselines.no_memory import NoMemoryBackend
from src.memory_baselines.shared_vector import SharedVectorMemoryBackend


METHOD_ALIASES = {
    "B0": "B0",
    "no_memory": "B0",
    "B1": "B1",
    "shared_vector": "B1",
    "B2": "B2",
    "isolated_vector": "B2",
    "isolated_memory": "B2",
    "per_agent_isolated": "B2",
    "B3": "B3",
    "gmemory": "B3",
    "g-memory": "B3",
    "gmemory_graph": "B3",
    "B4": "B4",
    "collaborative_memory": "B4",
    "access_controlled_memory": "B4",
    "private_shared_memory": "B4",
    "B5": "B5",
    "memlineage": "B5",
    "memlineage_adapted": "B5",
    "lineage_enforcement": "B5",
    "FlatProvenance": "FlatProvenance",
    "flat_provenance": "FlatProvenance",
    "flat-provenance": "FlatProvenance",
}


def normalize_method(method: str) -> str:
    try:
        return METHOD_ALIASES[method]
    except KeyError as exc:
        supported = ", ".join(sorted(METHOD_ALIASES))
        raise ValueError(f"Unsupported baseline method {method!r}. Supported methods: {supported}") from exc


def load_tasks(tasks_path: str, task_id: str | None = None) -> list[dict]:
    path = Path(tasks_path)
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.glob("*.json"))

    tasks = []
    for file in files:
        with file.open("r", encoding="utf-8") as f:
            task = json.load(f)
        if task_id is None or task.get("task_id") == task_id:
            tasks.append(task)

    if not tasks:
        raise ValueError(f"No tasks found in {tasks_path!r}" + (f" for task_id={task_id!r}" if task_id else ""))

    return tasks


def build_memory_backend(method: str):
    method = normalize_method(method)
    if method == "B0":
        return NoMemoryBackend()
    if method == "B1":
        return SharedVectorMemoryBackend()
    if method == "B2":
        return PerAgentIsolatedMemoryBackend()
    if method == "B3":
        return GMemoryGraphBackend()
    if method == "B4":
        return AccessControlledMemoryBackend()
    if method == "B5":
        return LineageEnforcementMemoryBackend()
    if method == "FlatProvenance":
        return FlatProvenanceMemoryBackend()

    raise ValueError(
        f"Unsupported baseline method {method!r}. "
        "Currently implemented: B0, B1, B2, B3, B4, B5, "
        "FlatProvenance."
    )


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


def output_paths(output_dir: str | Path, method: str, run_id: str | None, task_id: str) -> dict[str, Path]:
    output_root = Path(output_dir)

    def method_dir(kind: str) -> Path:
        root = output_root / kind / method
        return root / run_id if run_id else root

    paths = {
        "trace": method_dir("traces") / f"{task_id}.json",
        "memory_log": method_dir("memory_logs") / f"{task_id}.json",
        "decision_log": method_dir("decisions") / f"{task_id}.json",
    }
    if method in {"B3", "B4"}:
        graph_dir = output_root / "map_graphs" / method
        if run_id:
            graph_dir = graph_dir / run_id
        paths["graph"] = graph_dir / f"{task_id}.json"
    return paths


def save_baseline_outputs(result: dict, memory_backend, output_dir: str | Path) -> None:
    save_workflow_outputs(result, output_dir)
    exporter = getattr(memory_backend, "export_graph", None)
    if not callable(exporter):
        return
    graph = exporter()
    output_root = Path(output_dir)
    run_id = result.get("trace", {}).get("run_id")
    graph_dir = output_root / "map_graphs" / result["method"]
    if run_id:
        graph_dir = graph_dir / run_id
    graph_dir.mkdir(parents=True, exist_ok=True)
    with (graph_dir / f"{result['task_id']}.json").open("w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)


def task_outputs_exist(output_dir: str | Path, method: str, run_id: str | None, task_id: str) -> bool:
    paths = output_paths(output_dir, method, run_id, task_id)
    return all(path.exists() for path in paths.values())


def rehydrate_gmemory_task(
    memory_backend: GMemoryGraphBackend,
    task: dict,
    output_dir: str | Path,
    run_id: str | None,
) -> None:
    """Rebuild B3's process-local three-graph history from saved artifacts."""
    task_id = str(task.get("task_id", "unknown_task"))
    paths = output_paths(output_dir, "B3", run_id, task_id)
    with paths["trace"].open("r", encoding="utf-8") as f:
        trace = json.load(f)
    with paths["decision_log"].open("r", encoding="utf-8") as f:
        decision_log = json.load(f)
    with paths["graph"].open("r", encoding="utf-8") as f:
        graph_artifact = json.load(f)

    events = list(trace.get("events") or [])
    decisions = list(decision_log.get("decisions") or [])
    conversation = [
        {
            "agent_id": event.get("agent_id"),
            "content": event.get("content", ""),
            "visibility": "shared",
            "permission_scope": list(task.get("agent_roles") or []),
        }
        for event in events
        if event.get("event_type") == "message"
    ]
    runtime_task = sanitize_runtime_task(task)
    memory_backend.reset_task(task_id)
    # Rehydration reconstructs graph state from already-produced artifacts; it
    # must not issue fresh G-Memory LLM calls or charge them to the next task.
    previous_llm_augmentation = memory_backend.llm_augmentation
    previous_insight_llm_augmentation = (
        memory_backend.insight_llm_augmentation
    )
    memory_backend.llm_augmentation = False
    memory_backend.insight_llm_augmentation = False
    try:
        memory_backend.finalize_task(
            runtime_task,
            events,
            decisions,
            conversation,
        )
        insight_state = graph_artifact.get("insight_state")
        if isinstance(insight_state, list):
            memory_backend.restore_insight_state(insight_state)
    finally:
        memory_backend.llm_augmentation = previous_llm_augmentation
        memory_backend.insight_llm_augmentation = (
            previous_insight_llm_augmentation
        )
        memory_backend.drain_model_call_stats()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=sorted(METHOD_ALIASES))
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
    method = normalize_method(args.method)

    tasks = load_tasks(args.tasks, task_id=args.task_id)
    run_ids = [args.run_id] if args.run_id else [f"{args.run_prefix}{i}" for i in range(1, args.runs + 1)]
    all_completed = {}
    total_tasks = len(tasks)
    total_work = len(run_ids) * total_tasks
    finished_work = 0
    experiment_start = time.perf_counter()

    log_progress(
        "[baseline:start] "
        f"method={method} runs={len(run_ids)} tasks_per_run={total_tasks} total_tasks={total_work}"
    )

    for run_index, run_id in enumerate(run_ids, start=1):
        completed = []
        memory_backend = build_memory_backend(method)
        run_start = time.perf_counter()
        log_progress(
            "[baseline:run_start] "
            f"method={method} run={run_id} run_index={run_index}/{len(run_ids)} tasks={total_tasks}"
        )
        for task_index, task in enumerate(tasks, start=1):
            task_id = str(task.get("task_id", "unknown_task"))
            if args.skip_existing and task_outputs_exist(args.output_dir, method, run_id, task_id):
                if method == "B3":
                    rehydrate_gmemory_task(memory_backend, task, args.output_dir, run_id)
                    log_progress(
                        "[baseline:rehydrate] "
                        f"method={method} run={run_id} task={task_index}/{total_tasks} task_id={task_id}"
                    )
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
                remaining_total = total_work - finished_work
                eta_total = avg_total * remaining_total
                remaining_run = total_tasks - task_index
                eta_run = (run_elapsed / task_index) * remaining_run if task_index else 0
                log_progress(
                    "[baseline:skip] "
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
            task_elapsed = time.perf_counter() - task_start
            result["trace"]["run_id"] = run_id
            result["memory_log"]["run_id"] = run_id
            result["decision_log"]["run_id"] = run_id
            save_baseline_outputs(result, memory_backend, args.output_dir)
            finished_work += 1
            run_elapsed = time.perf_counter() - run_start
            total_elapsed = time.perf_counter() - experiment_start
            avg_total = total_elapsed / finished_work
            remaining_total = total_work - finished_work
            eta_total = avg_total * remaining_total
            remaining_run = total_tasks - task_index
            eta_run = (run_elapsed / task_index) * remaining_run if task_index else 0
            decisions = result["decision_log"]["decisions"]
            completed.append(
                {
                    "task_id": result["task_id"],
                    "domain": result["domain"],
                    "method": result["method"],
                    "decisions": len(result["decision_log"]["decisions"]),
                }
            )
            log_progress(
                "[baseline:task] "
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
            "[baseline:run_done] "
            f"method={method} run={run_id} completed={len(completed)}/{total_tasks} "
            f"run_elapsed={format_duration(time.perf_counter() - run_start)}"
        )

    log_progress(
        "[baseline:done] "
        f"method={method} runs={len(run_ids)} completed={finished_work}/{total_work} "
        f"elapsed={format_duration(time.perf_counter() - experiment_start)}"
    )
    print(json.dumps({"method": method, "run_ids": run_ids, "completed": all_completed}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
