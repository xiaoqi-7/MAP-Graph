# MAP-Graph final experiment runbook

This runbook describes the implementation and artifacts used by the submitted
paper. Historical smoke runs and superseded B4/B5 outputs are not canonical.

## 1. Canonical data and result sources

- Full task set: `data/final_tasks/` (2,700 full task specifications;
  evaluator-only fields are stripped before model/backend access)
- Evaluator ground truth: `data/final_ground_truth/`
- Transfer subset: `data/backbone_transfer_540/` (540 tasks)
- Transfer manifest: `data/backbone_transfer_540_manifest.json`
- B0--B3 and MAP-Graph main artifacts: `outputs_final_paper/`
- Final B4, B5, and B6 artifacts: `outputs_qwen_repro_current_full/`
- Ablation artifacts: `outputs_final_paper_ablation/`
- GLM transfer artifacts: `outputs_backbone_glm4_9b/`
- Llama transfer artifacts: `outputs_backbone_llama31_8b_openrouter/`
- Paper-facing merged metrics: `final_results/canonical/`

The main result table intentionally combines B0--B3 and MAP-Graph from
`outputs_final_paper/` with B4, B5, and FlatProvenance/B6 from
`outputs_qwen_repro_current_full/`. Do not substitute the superseded B4/B5
artifacts that were formerly stored in `outputs_final_paper/`.

The 540-task subset is a deterministic stratified sample of the 2,700 final
tasks. Seed 2027 chooses members within domain × experiment group × oracle
label strata; it does not generate new task content.

## 2. Methods

Main experiment:

- B0: no persistent memory
- B1: shared vector memory
- B2: per-agent isolated vector memory
- B3: G-Memory core-mechanism adaptation
- B4: Collaborative Memory core-mechanism adaptation
- B5: MemLineage core-mechanism adaptation
- B6: Flat Provenance control (`FlatProvenance` in code)
- MAPGraph: proposed method

Backbone transfer compares B1, B4, B5, B6, and MAPGraph. B6 retains direct
visibility, owner, permission, source-trust, private/revoked/poisoned, and
action-risk metadata, but constructs no provenance graph and performs no
recursive ancestry traversal or path-trust propagation.

B3--B5 are benchmark adaptations, not complete reproductions of the external
systems. The exact evaluated mechanisms are documented in the supplementary
material and source comments.

## 3. Environment

Install the pinned Python dependencies:

```bash
conda create -n mapgraph python=3.12 -y
conda run -n mapgraph pip install -r requirements.txt
```

Copy `.env.example` to a local `.env` and insert your own API key. Never
commit or upload `.env`, `.env.key*`, or `.envllama`.

Main Qwen settings:

```text
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
LLM_MODEL_FAMILY=qwen
LLM_TEMPERATURE=0
LLM_MAX_TOKENS=2048
LLM_TIMEOUT=120
LLM_MAX_RETRIES=5
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_DEVICE=cpu
EMBEDDING_NORMALIZE=true
EMBEDDING_LOCAL_FILES_ONLY=false
```

The reference environment is Python 3.12 with CPU embeddings and remote
OpenAI-compatible inference.  The package contains neither model weights nor
API keys.  `EMBEDDING_LOCAL_FILES_ONLY=false` is required on a fresh machine
to download `BAAI/bge-small-en-v1.5`; after that model is cached, it may be set
to `true` for offline embedding startup.  The package itself is approximately
54 MB.  Retained raw outputs require several gigabytes and are not included.
A full main/ablation/transfer rerun is API-intensive and long-running; exact
wall time depends on provider capacity and queueing.

The reported B3 adaptation used the following explicit settings:

```text
GMEMORY_STRICT_LLM=false
GMEMORY_INSIGHT_LLM=true
GMEMORY_MAX_GRAPH_TASKS=512
GMEMORY_START_INSIGHTS_THRESHOLD=5
GMEMORY_ROUNDS_PER_INSIGHTS=5
GMEMORY_INSIGHTS_POINT_NUM=5
```

Thus B3 uses deterministic key-step extraction and similarity reranking,
while retaining periodic LLM insight updates and a bounded ten-rule insight
graph.  These values are already present in `.env.example`.

Preflight:

```bash
conda run -n mapgraph python scripts/test_embedding.py
conda run -n mapgraph python scripts/test_api.py
conda run -n mapgraph python scripts/test_autogen_client.py
```

## 4. Dataset validation

```bash
conda run -n mapgraph python scripts/build_dataset_large.py --validate-only
conda run -n mapgraph python scripts/build_backbone_transfer_subset.py
```

The first command checks that the generator reproduces `final_tasks/` and
`final_ground_truth/` with seed 42. The second reconstructs the fixed
540-task transfer subset.

## 5. Main experiment

Run a baseline:

```bash
conda run -n mapgraph python scripts/run_baseline.py \
  --method B1 \
  --tasks data/final_tasks \
  --output-dir outputs_new_main \
  --run-id run1 \
  --max-rounds 1 \
  --model-max-tokens 2048 \
  --model-timeout 120 \
  --model-max-retries 5 \
  --skip-existing
```

Valid baseline method names are B0, B1, B2, B3, B4, B5, and
FlatProvenance. Use FlatProvenance for paper label B6.

Run MAP-Graph:

```bash
conda run -n mapgraph python scripts/run_mapgraph.py \
  --tasks data/final_tasks \
  --output-dir outputs_new_main \
  --run-id run1 \
  --max-rounds 1 \
  --model-max-tokens 2048 \
  --model-timeout 120 \
  --model-max-retries 5 \
  --skip-existing
```

Evaluate a complete output root:

```bash
conda run -n mapgraph python scripts/evaluate.py \
  --outputs outputs_new_main \
  --tasks data/final_tasks \
  --results-dir results_new_main \
  --run-id run1 \
  --bootstrap-samples 2000
```

The retained main outputs were split across two roots as documented in
Section 1.  Those multi-gigabyte raw roots are intentionally not included in
the supplementary package, so the canonical builder cannot be run directly
against the unpacked archive.  The supplied canonical/per-task tables can be
checked without raw traces by:

```bash
conda run -n mapgraph python scripts/verify_canonical_results.py
```

To rebuild rather than verify them, first rerun all methods into new complete
output roots, evaluate those roots, and update the source paths in
`scripts/build_canonical_results.py` before invoking it.  Do not treat the
provenance names in Section 1 as paths bundled in this archive.

## 6. Ablations

The six paper variants are:

```text
compressed_graph_only
wo_action_time_gate
wo_contamination_containment
wo_permission_filter
wo_provenance_graph
wo_trust_propagation
```

Run one variant:

```bash
conda run -n mapgraph python scripts/run_ablation.py \
  --variant wo_trust_propagation \
  --tasks data/final_tasks \
  --output-dir outputs_new_ablation \
  --run-id run1 \
  --max-rounds 1 \
  --model-max-tokens 2048 \
  --model-timeout 120 \
  --model-max-retries 5 \
  --skip-existing
```

Evaluate:

```bash
conda run -n mapgraph python scripts/evaluate_ablation.py \
  --main-outputs outputs_new_main \
  --ablation-outputs outputs_new_ablation \
  --tasks data/final_tasks \
  --results-dir results_new_ablation \
  --run-id run1
```

## 7. Backbone transfer

Only the generation backbone changes. Tasks, prompts, roles, embedding model,
temperature, rounds, token cap, retrieval settings, and evaluator remain
fixed.

Retained backbones:

- `Qwen/Qwen2.5-7B-Instruct`
- `THUDM/GLM-4-9B-0414`
- `meta-llama/llama-3.1-8b-instruct`

The GLM and Llama experiments each run B1, B4, B5, FlatProvenance, and
MAPGraph once on `data/backbone_transfer_540/`. Qwen transfer values are
extracted from the same 540 task IDs in the final full runs.

## 8. Integrity checks

```bash
sha256sum -c configs/reported_main_run_b4_b5.sha256
sha256sum -c configs/backbone_transfer_code.sha256
conda run -n mapgraph python -m unittest tests.test_paper_baselines
conda run -n mapgraph python scripts/build_dataset_large.py --validate-only
conda run -n mapgraph python scripts/verify_canonical_results.py
conda run -n mapgraph python scripts/verify_package.py
```

Do not write new experiments into any retained final output directory. Use a
new output root for every new run.
