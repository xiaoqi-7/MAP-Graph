#!/usr/bin/env python3
"""Run MAP-Graph component ablations without overwriting main experiment outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.autogen_workflow import AutoGenWorkflow, WorkflowConfig
from src.map_graph import ABLATION_PRESETS, MAPGraphMemoryBackend, ablation_config_from_name

from scripts.run_mapgraph import (
    decision_summary,
    format_duration,
    load_tasks,
    log_progress,
    save_map_graph_outputs,
    task_outputs_exist,
)


DOCUMENT_ABLATION_VARIANTS = [
    "wo_provenance_graph",
    "wo_permission_filter",
    "wo_trust_propagation",
    "wo_action_time_gate",
    "wo_contamination_containment",
    "compressed_graph_only",
]


def variant_names(selected: list[str]) -> list[str]:
    if not selected or selected == ["all"]:
        return list(DOCUMENT_ABLATION_VARIANTS)
    names: list[str] = []
    for name in selected:
        if name == "all":
            for preset in DOCUMENT_ABLATION_VARIANTS:
                if preset not in names:
                    names.append(preset)
            continue
        ablation_config_from_name(name)
        if name not in names:
            names.append(name)
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MAP-Graph ablation variants.")
    parser.add_argument("--tasks", default="data/final_tasks")
    parser.add_argument("--output-dir", default="outputs_final_paper_ablation")
    parser.add_argument(
        "--variant",
        "--ablation",
        action="append",
        default=[],
        dest="variant",
        help="Ablation variant. Repeatable. Use 'all' for all variants. Alias: --ablation.",
    )
    parser.add_argument("--list-variants", action="store_true")
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

    if args.list_variants:
        print(json.dumps({name: config.to_dict() for name, config in ABLATION_PRESETS.items()}, ensure_ascii=False, indent=2))
        return

    tasks = load_tasks(args.tasks, task_id=args.task_id)
    variants = variant_names(args.variant)
    run_ids = [args.run_id] if args.run_id else [f"{args.run_prefix}{i}" for i in range(1, args.runs + 1)]
    total_tasks = len(tasks)
    total_work = len(variants) * len(run_ids) * total_tasks
    finished_work = 0
    experiment_start = time.perf_counter()
    all_completed: dict[str, dict[str, list[dict]]] = {}

    log_progress(
        "[ablation:start] "
        f"variants={len(variants)} runs={len(run_ids)} tasks_per_run={total_tasks} total_tasks={total_work} "
        f"output_dir={args.output_dir}"
    )

    for variant_index, variant in enumerate(variants, start=1):
        config = ablation_config_from_name(variant)
        method = config.method_name
        all_completed[method] = {}
        log_progress(
            "[ablation:variant_start] "
            f"variant={variant} method={method} variant_index={variant_index}/{len(variants)}"
        )

        for run_index, run_id in enumerate(run_ids, start=1):
            completed = []
            memory_backend = MAPGraphMemoryBackend(ablation=config)
            run_start = time.perf_counter()
            log_progress(
                "[ablation:run_start] "
                f"variant={variant} method={method} run={run_id} run_index={run_index}/{len(run_ids)} tasks={total_tasks}"
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
                            "variant": variant,
                            "decisions": "skipped_existing",
                        }
                    )
                    run_elapsed = time.perf_counter() - run_start
                    total_elapsed = time.perf_counter() - experiment_start
                    avg_total = total_elapsed / max(1, finished_work)
                    eta_total = avg_total * (total_work - finished_work)
                    eta_run = (run_elapsed / task_index) * (total_tasks - task_index) if task_index else 0
                    log_progress(
                        "[ablation:skip] "
                        f"variant={variant} method={method} run={run_id} run_index={run_index}/{len(run_ids)} "
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
                result["trace"]["ablation"] = config.to_dict()
                result["memory_log"]["ablation"] = config.to_dict()
                result["decision_log"]["ablation"] = config.to_dict()
                run_config = dict(result["trace"].get("run_config") or {})
                run_config.update(
                    {
                        "run_id": run_id,
                        "variant": variant,
                        "retrieval_top_k": memory_backend.top_k,
                    }
                )
                result["trace"]["run_config"] = run_config
                result["memory_log"]["run_config"] = run_config
                result["decision_log"]["run_config"] = run_config
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
                        "variant": variant,
                        "decisions": len(decisions),
                    }
                )
                log_progress(
                    "[ablation:task] "
                    f"variant={variant} method={method} run={run_id} run_index={run_index}/{len(run_ids)} "
                    f"task={task_index}/{total_tasks} total={finished_work}/{total_work} "
                    f"task_id={result['task_id']} domain={result['domain']} "
                    f"decisions={decision_summary(decisions)} "
                    f"task_time={format_duration(task_elapsed)} "
                    f"run_elapsed={format_duration(run_elapsed)} run_eta={format_duration(eta_run)} "
                    f"total_elapsed={format_duration(total_elapsed)} total_eta={format_duration(eta_total)}"
                )

            all_completed[method][run_id] = completed
            log_progress(
                "[ablation:run_done] "
                f"variant={variant} method={method} run={run_id} completed={len(completed)}/{total_tasks} "
                f"run_elapsed={format_duration(time.perf_counter() - run_start)}"
            )

    log_progress(
        "[ablation:done] "
        f"variants={len(variants)} runs={len(run_ids)} completed={finished_work}/{total_work} "
        f"elapsed={format_duration(time.perf_counter() - experiment_start)}"
    )
    print(
        json.dumps(
            {
                "variants": variants,
                "methods": [ablation_config_from_name(name).method_name for name in variants],
                "run_ids": run_ids,
                "completed": all_completed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
