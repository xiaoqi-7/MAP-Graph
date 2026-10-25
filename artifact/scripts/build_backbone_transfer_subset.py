#!/usr/bin/env python3
"""Build a deterministic stratified subset for backbone-transfer experiments."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import shutil
from typing import Any


STRATUM_FIELDS = (
    "domain",
    "experiment_group",
    "ground_truth_decision",
)


def load_tasks(source: Path) -> list[tuple[Path, dict[str, Any]]]:
    files = sorted(source.glob("*.json"))
    tasks: list[tuple[Path, dict[str, Any]]] = []
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            task = json.load(handle)
        missing = [field for field in STRATUM_FIELDS if not task.get(field)]
        if missing:
            raise ValueError(
                f"{path} is missing required stratum fields: {missing}"
            )
        if not task.get("task_id"):
            raise ValueError(f"{path} is missing task_id")
        tasks.append((path, task))
    if not tasks:
        raise ValueError(f"No task JSON files found under {source}")
    return tasks


def stratum(task: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(task[field]) for field in STRATUM_FIELDS)


def allocate_quotas(
    strata: dict[tuple[str, ...], list[tuple[Path, dict[str, Any]]]],
    subset_size: int,
    total_size: int,
) -> dict[tuple[str, ...], int]:
    if subset_size < len(strata):
        raise ValueError(
            f"subset size {subset_size} cannot cover all {len(strata)} strata"
        )
    if subset_size > total_size:
        raise ValueError(
            f"subset size {subset_size} exceeds source size {total_size}"
        )

    exact = {
        key: subset_size * len(items) / total_size
        for key, items in strata.items()
    }
    quotas = {
        key: min(len(strata[key]), max(1, int(value)))
        for key, value in exact.items()
    }

    while sum(quotas.values()) < subset_size:
        candidates = [
            key
            for key in strata
            if quotas[key] < len(strata[key])
        ]
        if not candidates:
            raise RuntimeError("Unable to allocate the requested subset size")
        key = max(
            candidates,
            key=lambda item: (
                exact[item] - quotas[item],
                len(strata[item]),
                item,
            ),
        )
        quotas[key] += 1

    while sum(quotas.values()) > subset_size:
        candidates = [key for key in strata if quotas[key] > 1]
        if not candidates:
            raise RuntimeError("Unable to reduce quotas to the requested size")
        key = max(
            candidates,
            key=lambda item: (
                quotas[item] - exact[item],
                quotas[item],
                item,
            ),
        )
        quotas[key] -= 1

    return quotas


def stable_seed(seed: int, key: tuple[str, ...]) -> int:
    payload = f"{seed}|" + "|".join(key)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def nested_counts(
    tasks: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for field in STRATUM_FIELDS:
        result[field] = dict(
            sorted(Counter(str(task[field]) for task in tasks).items())
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sample tasks by domain, experiment_group, and "
            "ground_truth_decision."
        )
    )
    parser.add_argument("--source", default="data/final_tasks")
    parser.add_argument("--out", default="data/backbone_transfer_540")
    parser.add_argument("--size", type=int, default=540)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--manifest",
        default=None,
        help="Manifest path outside the task directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing task JSON files in --out.",
    )
    args = parser.parse_args()

    source = Path(args.source)
    destination = Path(args.out)
    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else destination.parent / f"{destination.name}_manifest.json"
    )
    loaded = load_tasks(source)

    grouped: dict[
        tuple[str, ...],
        list[tuple[Path, dict[str, Any]]],
    ] = defaultdict(list)
    for item in loaded:
        grouped[stratum(item[1])].append(item)
    quotas = allocate_quotas(grouped, args.size, len(loaded))

    selected: list[tuple[Path, dict[str, Any]]] = []
    for key in sorted(grouped):
        candidates = sorted(
            grouped[key],
            key=lambda item: str(item[1]["task_id"]),
        )
        rng = random.Random(stable_seed(args.seed, key))
        selected.extend(rng.sample(candidates, quotas[key]))
    selected.sort(key=lambda item: str(item[1]["task_id"]))

    existing = sorted(destination.glob("*.json"))
    if existing and not args.overwrite:
        raise FileExistsError(
            f"{destination} already contains {len(existing)} JSON files; "
            "pass --overwrite to replace them"
        )
    destination.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in existing:
            path.unlink()
    for source_path, _ in selected:
        shutil.copy2(source_path, destination / source_path.name)

    selected_tasks = [task for _, task in selected]
    selected_ids = [str(task["task_id"]) for task in selected_tasks]
    id_digest = hashlib.sha256(
        ("\n".join(selected_ids) + "\n").encode("utf-8")
    ).hexdigest()
    stratum_counts = {
        "|".join(key): quotas[key]
        for key in sorted(quotas)
    }
    manifest = {
        "source": str(source),
        "destination": str(destination),
        "source_size": len(loaded),
        "subset_size": len(selected),
        "seed": args.seed,
        "stratum_fields": list(STRATUM_FIELDS),
        "num_strata": len(grouped),
        "task_id_sha256": id_digest,
        "counts": nested_counts(selected_tasks),
        "stratum_counts": stratum_counts,
        "selected_task_ids": selected_ids,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
