from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_ROOT = PROJECT_ROOT / "final_results"
CANONICAL_ROOT = FINAL_ROOT / "canonical"

LEGACY_METRICS = FINAL_ROOT / "evaluated" / "main_legacy" / "metrics"
CURRENT_METRICS = FINAL_ROOT / "evaluated" / "main_current" / "metrics"
LLAMA_METRICS = FINAL_ROOT / "evaluated" / "backbone_llama" / "metrics"
GLM_METRICS = PROJECT_ROOT / "outputs_backbone_glm4_9b" / "metrics"
ABLATION_METRICS = PROJECT_ROOT / "outputs_final_paper_ablation" / "metrics"

LEGACY_METHODS = {"B0", "B1", "B2", "B3", "MAPGraph"}
CURRENT_METHODS = {"B4", "B5", "FlatProvenance"}
METHOD_ORDER = ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "MAPGraph"]
TRANSFER_ORDER = ["B1", "B4", "B5", "B6", "MAPGraph"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames = fields or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def number(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value in (None, "", "nan", "NaN"):
        return float("nan")
    return float(value)


def percent(value: float) -> float:
    return round(100.0 * value, 2)


def display_method(method: str) -> str:
    return "B6" if method == "FlatProvenance" else method


def selected_main_rows(filename: str) -> list[dict[str, str]]:
    old_rows = [
        row
        for row in read_csv(LEGACY_METRICS / filename)
        if row["method"] in LEGACY_METHODS
    ]
    current_rows = [
        row
        for row in read_csv(CURRENT_METRICS / filename)
        if row["method"] in CURRENT_METHODS
    ]
    rows = old_rows + current_rows
    for row in rows:
        row["method"] = display_method(row["method"])
    order = {method: index for index, method in enumerate(METHOD_ORDER)}
    return sorted(rows, key=lambda row: (order[row["method"]], row.get("task_id", "")))


def selected_main_ci_rows() -> list[dict[str, str]]:
    old_path = (
        FINAL_ROOT
        / "evaluated"
        / "main_legacy"
        / "tables"
        / "tables"
        / "main_paper_results.csv"
    )
    current_path = (
        FINAL_ROOT
        / "evaluated"
        / "main_current"
        / "tables"
        / "tables"
        / "main_paper_results.csv"
    )
    rows = [
        row for row in read_csv(old_path) if row["method"] in LEGACY_METHODS
    ] + [
        row for row in read_csv(current_path) if row["method"] in CURRENT_METHODS
    ]
    for row in rows:
        row["method"] = display_method(row["method"])
    order = {method: index for index, method in enumerate(METHOD_ORDER)}
    return sorted(rows, key=lambda row: order[row["method"]])


def aggregate_rate(rows: list[dict[str, str]], field: str) -> float:
    return mean(number(row, field) for row in rows)


def main_paper_table(summary_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    result = []
    for row in summary_rows:
        uacc = row.get("unauthorized_access_rate", "")
        result.append(
            {
                "method": row["method"],
                "n": int(float(row["n"])),
                "TSR_percent": percent(number(row, "task_success_rate")),
                "Acc_percent": percent(number(row, "decision_accuracy")),
                "Clean_percent": percent(number(row, "clean_task_success_rate")),
                "Unsafe_percent": percent(number(row, "unsafe_action_rate")),
                "ASR_percent": percent(number(row, "attack_success_rate")),
                "Leakage_percent": percent(number(row, "privacy_leakage_rate")),
                "UAcc_percent": "" if uacc in ("", "nan", "NaN") else percent(float(uacc)),
                "Revocation_percent": percent(number(row, "revocation_violation_rate")),
            }
        )
    return result


def scenario_table(task_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    result = []
    groups = sorted({row["experiment_group"] for row in task_rows})
    for group in groups:
        by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in task_rows:
            if row["experiment_group"] == group:
                by_method[row["method"]].append(row)
        for method in METHOD_ORDER:
            rows = by_method[method]
            result.append(
                {
                    "experiment_group": group,
                    "method": method,
                    "n": len(rows),
                    "TSR_percent": percent(aggregate_rate(rows, "task_success")),
                    "Acc_percent": percent(aggregate_rate(rows, "decision_correct")),
                    "Unsafe_percent": percent(aggregate_rate(rows, "unsafe_action")),
                    "ASR_percent": percent(aggregate_rate(rows, "attack_success")),
                    "Leakage_percent": percent(aggregate_rate(rows, "privacy_leakage")),
                    "Revocation_percent": percent(aggregate_rate(rows, "revocation_violation")),
                }
            )
    return result


def domain_table(task_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    map_rows = [row for row in task_rows if row["method"] == "MAPGraph"]
    result = []
    for domain in sorted({row["domain"] for row in map_rows}):
        rows = [row for row in map_rows if row["domain"] == domain]
        result.append(
            {
                "method": "MAPGraph",
                "domain": domain,
                "n": len(rows),
                "TSR_percent": percent(aggregate_rate(rows, "task_success")),
                "Acc_percent": percent(aggregate_rate(rows, "decision_correct")),
            }
        )
    return result


def cap_rate(method: str) -> tuple[int, int, float]:
    stored_method = "FlatProvenance" if method == "B6" else method
    output_root = (
        PROJECT_ROOT / "outputs_qwen_repro_current_full"
        if method in {"B4", "B5", "B6"}
        else PROJECT_ROOT / "outputs_final_paper"
    )
    calls = 0
    capped = 0
    for path in sorted((output_root / "traces" / stored_method / "run1").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for event in payload.get("events") or []:
            model_call = event.get("model_call") or {}
            if "completion_tokens" not in model_call:
                continue
            calls += 1
            capped += int(float(model_call["completion_tokens"]) >= 2048)
    if calls != 10_800:
        raise ValueError(f"{method}: expected 10,800 outer calls, found {calls}")
    return capped, calls, percent(capped / calls)


def diagnostics_table(summary_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    result = []
    for row in summary_rows:
        capped, calls, cap_percent = cap_rate(row["method"])
        result.append(
            {
                "method": row["method"],
                "block_rate_percent": percent(number(row, "block_rate")),
                "correct_block_percent": percent(number(row, "correct_block_rate")),
                "false_block_percent": percent(number(row, "false_block_rate")),
                "structural_contamination_percent": percent(
                    number(row, "contamination_propagation_rate")
                ),
                "avg_total_tokens": round(number(row, "avg_total_tokens"), 1),
                "capped_outer_calls": capped,
                "outer_calls": calls,
                "cap_rate_percent": cap_percent,
                "avg_graph_nodes": round(number(row, "avg_graph_nodes"), 2),
                "avg_graph_edges": round(number(row, "avg_graph_edges"), 2),
            }
        )
    return result


def subset_ids() -> set[str]:
    return {path.stem for path in (PROJECT_ROOT / "data" / "backbone_transfer_540").glob("*.json")}


def qwen_transfer_rows(main_task_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    ids = subset_ids()
    return [
        row
        for row in main_task_rows
        if row["method"] in TRANSFER_ORDER and row["task_id"] in ids
    ]


def transfer_summary_from_tasks(
    backbone: str, rows: list[dict[str, str]]
) -> list[dict[str, Any]]:
    result = []
    for method in TRANSFER_ORDER:
        selected = [row for row in rows if display_method(row["method"]) == method]
        if len(selected) != 540:
            raise ValueError(f"{backbone}/{method}: expected 540 tasks, found {len(selected)}")
        result.append(
            {
                "backbone": backbone,
                "method": method,
                "n": len(selected),
                "TSR_percent": percent(aggregate_rate(selected, "task_success")),
                "Acc_percent": percent(aggregate_rate(selected, "decision_correct")),
                "Unsafe_percent": percent(aggregate_rate(selected, "unsafe_action")),
            }
        )
    return result


def transfer_summary_from_aggregate(
    backbone: str, path: Path
) -> list[dict[str, Any]]:
    source = {display_method(row["method"]): row for row in read_csv(path)}
    result = []
    for method in TRANSFER_ORDER:
        row = source[method]
        result.append(
            {
                "backbone": backbone,
                "method": method,
                "n": int(float(row["n"])),
                "TSR_percent": percent(number(row, "task_success_rate")),
                "Acc_percent": percent(number(row, "decision_accuracy")),
                "Unsafe_percent": percent(number(row, "unsafe_action_rate")),
            }
        )
    return result


def uacc_table(backbone: str, task_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    result = []
    for method in TRANSFER_ORDER:
        rows = [row for row in task_rows if display_method(row["method"]) == method]
        unauthorized = sum(int(float(row["unauthorized_read_count"])) for row in rows)
        attempts = sum(int(float(row["unauthorized_read_attempt_count"])) for row in rows)
        result.append(
            {
                "backbone": backbone,
                "method": method,
                "unauthorized_reads": unauthorized,
                "observed_attempts": attempts,
                "UAcc_percent": "" if attempts == 0 else percent(unauthorized / attempts),
            }
        )
    return result


def ablation_table() -> list[dict[str, Any]]:
    rows = read_csv(ABLATION_METRICS / "ablation_summary_metrics.csv")
    names = {
        "MAPGraph": "Full",
        "MAPGraph_compressed_graph_only": "Compressed graph",
        "MAPGraph_wo_action_time_gate": "No action gate",
        "MAPGraph_wo_contamination_containment": "No containment",
        "MAPGraph_wo_permission_filter": "No permission filter",
        "MAPGraph_wo_provenance_graph": "No provenance graph",
        "MAPGraph_wo_trust_propagation": "No trust propagation",
    }
    result = []
    for row in rows:
        result.append(
            {
                "variant": names[row["method"]],
                "n": int(float(row["n"])),
                "TSR_percent": percent(number(row, "task_success_rate")),
                "Acc_percent": percent(number(row, "decision_accuracy")),
                "Clean_percent": percent(number(row, "clean_task_success_rate")),
                "Unsafe_percent": percent(number(row, "unsafe_action_rate")),
                "ASR_percent": percent(number(row, "attack_success_rate")),
                "Leakage_percent": percent(number(row, "privacy_leakage_rate")),
                "UAcc_percent": percent(number(row, "unauthorized_access_rate")),
                "Revocation_percent": percent(number(row, "revocation_violation_rate")),
                "contamination_percent": percent(
                    number(row, "contamination_propagation_rate")
                ),
                "block_percent": percent(number(row, "block_rate")),
                "correct_block_percent": percent(number(row, "correct_block_rate")),
                "false_block_percent": percent(number(row, "false_block_rate")),
                "avg_total_tokens": round(number(row, "avg_total_tokens"), 1),
                "avg_graph_nodes": round(number(row, "avg_graph_nodes"), 2),
                "avg_graph_edges": round(number(row, "avg_graph_edges"), 2),
            }
        )
    return result


def main() -> None:
    main_tasks = selected_main_rows("task_metrics.csv")
    main_summary = selected_main_rows("summary_metrics.csv")

    write_csv(CANONICAL_ROOT / "main" / "task_metrics.csv", main_tasks)
    write_csv(CANONICAL_ROOT / "main" / "summary_metrics.csv", main_summary)
    write_csv(
        CANONICAL_ROOT / "main" / "confidence_intervals.csv",
        selected_main_ci_rows(),
    )
    write_csv(CANONICAL_ROOT / "main" / "paper_table.csv", main_paper_table(main_summary))
    write_csv(CANONICAL_ROOT / "main" / "scenario_metrics.csv", scenario_table(main_tasks))
    write_csv(CANONICAL_ROOT / "main" / "mapgraph_domain_metrics.csv", domain_table(main_tasks))
    write_csv(CANONICAL_ROOT / "main" / "diagnostics.csv", diagnostics_table(main_summary))

    qwen_tasks = qwen_transfer_rows(main_tasks)
    write_csv(
        CANONICAL_ROOT / "backbone" / "qwen_task_metrics.csv",
        qwen_tasks,
    )
    glm_tasks = read_csv(GLM_METRICS / "task_metrics.csv")
    llama_tasks = read_csv(LLAMA_METRICS / "task_metrics.csv")
    transfer = (
        transfer_summary_from_tasks("Qwen2.5-7B-Instruct", qwen_tasks)
        + transfer_summary_from_aggregate(
            "GLM-4-9B-0414", GLM_METRICS / "summary_metrics.csv"
        )
        + transfer_summary_from_aggregate(
            "Llama-3.1-8B-Instruct", LLAMA_METRICS / "summary_metrics.csv"
        )
    )
    write_csv(CANONICAL_ROOT / "backbone" / "transfer_metrics.csv", transfer)
    write_csv(
        CANONICAL_ROOT / "backbone" / "uacc_diagnostics.csv",
        uacc_table("Qwen2.5-7B-Instruct", qwen_tasks)
        + uacc_table("GLM-4-9B-0414", glm_tasks)
        + uacc_table("Llama-3.1-8B-Instruct", llama_tasks),
    )
    write_csv(CANONICAL_ROOT / "ablation" / "metrics.csv", ablation_table())

    provenance = {
        "main": {
            "B0_B1_B2_B3_MAPGraph": "outputs_final_paper",
            "B4_B5_B6": "outputs_qwen_repro_current_full",
        },
        "ablation": "outputs_final_paper_ablation",
        "backbone": {
            "Qwen2.5-7B-Instruct": "stratified extraction from the final full runs",
            "GLM-4-9B-0414": "outputs_backbone_glm4_9b",
            "Llama-3.1-8B-Instruct": "outputs_backbone_llama31_8b_openrouter",
        },
        "method_alias": {"B6": "FlatProvenance"},
        "task_counts": {
            "main_per_method": 2700,
            "main_total": 21600,
            "ablation_per_variant": 2700,
            "backbone_per_method": 540,
        },
    }
    (CANONICAL_ROOT / "RESULT_PROVENANCE.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote canonical results to {CANONICAL_ROOT}")


if __name__ == "__main__":
    main()
