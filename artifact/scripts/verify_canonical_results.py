from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


ARTIFACT_ROOT = Path(__file__).resolve().parents[1]


def read_rows(relative_path: str) -> list[dict[str, str]]:
    path = ARTIFACT_ROOT / relative_path
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_per_task(
    relative_path: str,
    expected_methods: set[str],
    expected_count: int,
) -> set[str]:
    rows = read_rows(relative_path)
    task_ids_by_method: dict[str, set[str]] = defaultdict(set)
    seen: set[tuple[str, str]] = set()

    for row in rows:
        method = row.get("method", "")
        task_id = row.get("task_id", "")
        if not method or not task_id:
            raise ValueError(f"Missing method/task_id in {relative_path}")
        key = (method, task_id)
        if key in seen:
            raise ValueError(f"Duplicate method/task pair in {relative_path}: {key}")
        seen.add(key)
        task_ids_by_method[method].add(task_id)

    actual_methods = set(task_ids_by_method)
    if actual_methods != expected_methods:
        raise ValueError(
            f"Method mismatch in {relative_path}: "
            f"expected={sorted(expected_methods)}, actual={sorted(actual_methods)}"
        )
    for method, task_ids in sorted(task_ids_by_method.items()):
        if len(task_ids) != expected_count:
            raise ValueError(
                f"Coverage mismatch in {relative_path} for {method}: "
                f"expected={expected_count}, actual={len(task_ids)}"
            )

    reference = task_ids_by_method[next(iter(sorted(expected_methods)))]
    for method, task_ids in task_ids_by_method.items():
        if task_ids != reference:
            raise ValueError(f"Task-ID mismatch within {relative_path}: {method}")

    print(
        f"Verified {relative_path}: {len(expected_methods)} methods x "
        f"{expected_count} tasks = {len(rows)} rows."
    )
    return reference


def verify_aggregate(relative_path: str, expected_rows: int) -> None:
    rows = read_rows(relative_path)
    if len(rows) != expected_rows:
        raise ValueError(
            f"Row-count mismatch in {relative_path}: "
            f"expected={expected_rows}, actual={len(rows)}"
        )
    print(f"Verified {relative_path}: {len(rows)} aggregate rows.")


def main() -> None:
    main_methods = {"B0", "B1", "B2", "B3", "B4", "B5", "B6", "MAPGraph"}
    ablation_methods = {
        "MAPGraph",
        "MAPGraph_compressed_graph_only",
        "MAPGraph_wo_action_time_gate",
        "MAPGraph_wo_contamination_containment",
        "MAPGraph_wo_permission_filter",
        "MAPGraph_wo_provenance_graph",
        "MAPGraph_wo_trust_propagation",
    }
    transfer_qwen = {"B1", "B4", "B5", "B6", "MAPGraph"}
    transfer_other = {"B1", "B4", "B5", "FlatProvenance", "MAPGraph"}

    verify_per_task(
        "final_results/canonical/main/task_metrics.csv", main_methods, 2700
    )
    verify_per_task(
        "final_results/per_task/ablation_task_metrics.csv", ablation_methods, 2700
    )
    qwen_ids = verify_per_task(
        "final_results/canonical/backbone/qwen_task_metrics.csv", transfer_qwen, 540
    )
    glm_ids = verify_per_task(
        "final_results/per_task/glm_task_metrics.csv", transfer_other, 540
    )
    llama_ids = verify_per_task(
        "final_results/per_task/llama_task_metrics.csv", transfer_other, 540
    )
    if not (qwen_ids == glm_ids == llama_ids):
        raise ValueError("The three backbone evaluations do not use identical task IDs.")

    verify_aggregate("final_results/canonical/main/paper_table.csv", 8)
    verify_aggregate("final_results/canonical/ablation/metrics.csv", 7)
    verify_aggregate("final_results/canonical/backbone/transfer_metrics.csv", 15)
    verify_aggregate("final_results/canonical/backbone/uacc_diagnostics.csv", 15)
    print("Canonical result coverage is complete and internally aligned.")


if __name__ == "__main__":
    main()
