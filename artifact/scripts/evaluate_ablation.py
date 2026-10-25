#!/usr/bin/env python3
"""Evaluate MAP-Graph ablations against the full MAPGraph result."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate import (
    add_cluster_bootstrap_ci,
    aggregate,
    decision_files,
    failure_rows,
    load_tasks,
    paper_rows,
    task_metric_row,
    write_csv,
)


COMPARISON_METRICS = [
    "decision_accuracy",
    "answer_accuracy",
    "task_success_rate",
    "clean_task_success_rate",
    "unsafe_action_rate",
    "attack_success_rate",
    "privacy_leakage_rate",
    "unauthorized_access_rate",
    "contamination_propagation_rate",
    "revocation_violation_rate",
    "correct_block_rate",
    "false_block_rate",
    "avg_total_tokens",
    "avg_estimated_api_cost_usd",
    "allow_rate",
    "block_rate",
    "reverify_rate",
    "redact_rate",
    "ask_user_rate",
    "avg_graph_nodes",
    "avg_graph_edges",
    "avg_output_kb",
]


def read_rows(
    output_root: Path,
    tasks: dict[str, dict[str, Any]],
    method_prefix: str | None = None,
    run_id: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in decision_files(output_root):
        row = task_metric_row(output_root, path, tasks)
        if row is None:
            continue
        if method_prefix and not str(row["method"]).startswith(method_prefix):
            continue
        if run_id and str(row["run_id"]) != run_id:
            continue
        row["output_root"] = str(output_root)
        rows.append(row)
    return rows


def add_delta_vs_full(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    full = next((row for row in summary_rows if row.get("method") == "MAPGraph"), None)
    if full is None:
        return summary_rows

    out = []
    for row in summary_rows:
        enriched = dict(row)
        for metric in COMPARISON_METRICS:
            value = row.get(metric)
            full_value = full.get(metric)
            if isinstance(value, (int, float)) and isinstance(full_value, (int, float)):
                enriched[f"delta_vs_MAPGraph_{metric}"] = round(float(value) - float(full_value), 6)
        out.append(enriched)
    return out


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MAPGraph ablation outputs.")
    parser.add_argument("--main-outputs", default="outputs_final_paper", help="Output root containing the full MAPGraph run.")
    parser.add_argument("--ablation-outputs", default="outputs_final_paper_ablation", help="Output root containing ablation runs.")
    parser.add_argument("--tasks", default="data/final_tasks")
    parser.add_argument("--results-dir", default="results_final_paper")
    parser.add_argument("--run-id", default="run1")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--failure-limit", type=int, default=60000)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()

    tasks = load_tasks(Path(args.tasks))
    main_rows = [
        row
        for row in read_rows(Path(args.main_outputs), tasks, run_id=args.run_id)
        if row["method"] == "MAPGraph"
    ]
    ablation_rows = read_rows(
        Path(args.ablation_outputs),
        tasks,
        method_prefix="MAPGraph_",
        run_id=args.run_id,
    )
    rows = main_rows + ablation_rows
    if not rows:
        raise ValueError("No MAPGraph or ablation decision logs were found.")

    expected_methods = {
        "MAPGraph",
        "MAPGraph_wo_provenance_graph",
        "MAPGraph_wo_permission_filter",
        "MAPGraph_wo_trust_propagation",
        "MAPGraph_wo_action_time_gate",
        "MAPGraph_wo_contamination_containment",
        "MAPGraph_compressed_graph_only",
    }
    coverage = {
        method: sum(1 for row in rows if row["method"] == method)
        for method in sorted(expected_methods)
    }
    incomplete = {
        method: count
        for method, count in coverage.items()
        if count != len(tasks)
    }
    if incomplete and not args.allow_incomplete:
        details = ", ".join(f"{method}={count}/{len(tasks)}" for method, count in incomplete.items())
        raise ValueError(f"Incomplete ablation coverage for {args.run_id}: {details}")

    results_dir = Path(args.results_dir)
    tables_dir = results_dir / "tables"
    metrics_dir = Path(args.ablation_outputs) / "metrics"

    summary_rows = add_delta_vs_full(aggregate(rows, ["method"]))
    ablation_paper_rows = add_cluster_bootstrap_ci(
        paper_rows(summary_rows),
        rows,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    by_run_rows = aggregate(rows, ["method", "run_id"])
    by_experiment_rows = aggregate(rows, ["method", "experiment_group"])
    by_domain_experiment_rows = aggregate(rows, ["method", "domain", "experiment_group"])
    failures = failure_rows(rows, limit=args.failure_limit)

    write_csv(metrics_dir / "ablation_task_metrics.csv", rows)
    write_csv(metrics_dir / "ablation_summary_metrics.csv", summary_rows)
    write_csv(tables_dir / "ablation_results.csv", summary_rows)
    write_csv(tables_dir / "ablation_paper_results.csv", ablation_paper_rows)
    write_csv(tables_dir / "ablation_results_by_run.csv", by_run_rows)
    write_csv(tables_dir / "ablation_results_by_experiment.csv", by_experiment_rows)
    write_csv(tables_dir / "ablation_results_by_domain_experiment.csv", by_domain_experiment_rows)
    write_csv(tables_dir / "ablation_failure_cases.csv", failures)

    summary = {
        "num_task_specs": len(tasks),
        "num_full_mapgraph_rows": len(main_rows),
        "num_ablation_rows": len(ablation_rows),
        "methods": sorted({row["method"] for row in rows}),
        "runs": sorted({row["run_id"] for row in rows}),
        "selected_run": args.run_id,
        "coverage": coverage,
        "metrics_dir": str(metrics_dir),
        "tables_dir": str(tables_dir),
    }
    write_json(metrics_dir / "ablation_evaluation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
