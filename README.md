# MAP-Graph supplementary package

This package contains the anonymous supplementary paper, the paper-aligned
implementation, all synthetic task specifications, evaluator ground truth,
the fixed backbone-transfer subset, and the canonical per-task/result tables.
It contains no API keys, model weights, provider credentials, or private
human data.

## Contents

```text
Appendix.pdf

paper/
  supplementary-2027.tex
  anonymous-submission-latex-2026.tex
  references.bib
  MAP-Graph1.pdf
  MAP-Graph2.pdf
  aaai2027.sty
  aaai2027.bst

artifact/
  src/                         implementation
  scripts/                     generation, execution, and evaluation entry points
  tests/                       baseline/mechanism regression tests
  configs/                     model configuration and evaluated-code hashes
  data/final_tasks/            2,700 full task specifications; evaluator-only
                               fields are removed by the runtime sanitizer
  data/final_ground_truth/     evaluator-only oracle records
  data/backbone_transfer_540/  fixed stratified transfer subset
  final_results/canonical/     merged paper-facing metrics and confidence intervals
  final_results/per_task/      full ablation, GLM, and Llama evaluator rows
  requirements.txt
  .env.example
  EXPERIMENT_RUNBOOK.md
```

## Supplementary paper

Open `Appendix.pdf` for the compiled appendix, or compile
`paper/supplementary-2027.tex` in the official AAAI-27 Overleaf project.
The required `aaai2027.sty` and `aaai2027.bst` are included beside it. The
supplementary source is anonymous and uses the submission mode of the official
AAAI-27 style.

## Reproduction

From `artifact/`:

```bash
conda create -n mapgraph python=3.12 -y
conda run -n mapgraph pip install -r requirements.txt
cp .env.example .env
```

Insert a valid OpenAI-compatible API key in the local `.env`, then run:

```bash
conda run -n mapgraph python scripts/build_dataset_large.py --validate-only
conda run -n mapgraph python -m unittest tests.test_paper_baselines
```

Execution and evaluation commands are listed in
`artifact/EXPERIMENT_RUNBOOK.md`.

The reference environment uses Python 3.12, CPU sentence-transformer
embeddings, and remote OpenAI-compatible generation endpoints.  No model
weights or API keys are bundled.  On first use, leave
`EMBEDDING_LOCAL_FILES_ONLY=false` so the embedding model can be downloaded;
after it is cached, offline startup may be enabled.  The submitted package is
approximately 54 MB.  Full raw outputs occupy several gigabytes and are
omitted; a full rerun is API-intensive and long-running, with wall time
depending on provider capacity and queueing.

## Result provenance

The submitted main table combines:

- B0--B3 and MAP-Graph from the retained Qwen full run;
- B4, B5, and B6/FlatProvenance from the final Qwen reproduction run;
- the full ablation run;
- GLM-4-9B-0414 and Llama-3.1-8B-Instruct transfer runs;
- Qwen transfer values extracted on the identical 540 task identifiers.

Exact source-to-table mappings are recorded in
`artifact/final_results/canonical/RESULT_PROVENANCE.json`.

The five retained raw-output roots total approximately 3.2 GB uncompressed and
are not duplicated in this upload-oriented archive.  Keep them as a separate
anonymous full artifact for release if the venue permits.  This package
retains the complete
2,700-task main and ablation evaluator rows, all three 540-task transfer
evaluations, all aggregate diagnostics used in the paper, and the code needed
to rerun the experiments.

Because those raw roots are intentionally omitted,
`scripts/build_canonical_results.py` is not a self-contained archive command.
Use `scripts/verify_canonical_results.py` to check the supplied canonical and
per-task result coverage.  To rebuild the canonical tables, first rerun every
experiment into new output roots and then point the builder at those complete
roots.
