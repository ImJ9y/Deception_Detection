# Multilingual Deception Detection (Public Code Sample)

## Overview
This repository is a curated code sample from a multilingual machine learning pipeline for detecting AI-generated versus human-written reviews.

The sample includes:
- data extraction utility (`languages/extract_dataset.py`)
- LLM generation pipelines (`scripts/gen/`)
- staged evaluation orchestration (`scripts/eval/run_eval_pipeline.py`)
- judge, distributional, detector, and aggregation components (`scripts/eval/`)

## What This Demonstrates
- How I structure multi-stage ML workflows with explicit artifacts between stages
- How I separate generation, evaluation, aggregation, and reporting modules
- How I handle multilingual defaults, path resolution, and configurable scoring logic
- How I document assumptions, fallbacks, and reproducibility points

## Repository Layout
- `languages/extract_dataset.py`
- `scripts/gen/gen_qwen_reviews.py`
- `scripts/gen/gen_llama_reviews.py`
- `scripts/eval/run_eval_pipeline.py`
- `scripts/eval/dist/`
- `scripts/eval/judge/`
- `scripts/eval/detector/`
- `scripts/eval/collect_master_scoreboard.py`
- `scripts/eval/write_baseline_comparison.py`
- `configs/`

## Runnable Scope
- Runnable with local Python environment and dependencies.
- Generation steps require accessible model assets/checkpoints and compatible runtime.
- End-to-end runs require language CSVs under `languages/<language>/...`.

## Quick Start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas torch transformers
```

### Extract dataset to CSV
```bash
python3 languages/extract_dataset.py \
  --dataset "your_org/your_dataset" \
  --language "hausa" \
  --outpath "/absolute/path/languages/{language}/raw/real_{language}_reviews.csv"
```

### Generate synthetic reviews
```bash
python3 scripts/gen/gen_qwen_reviews.py \
  --targets_csv <targets_csv> \
  --run_id qwen_zs_hausa_v1 \
  --lang hausa \
  --prompt_config configs/gen_zs_config.json
```

```bash
python3 scripts/gen/gen_llama_reviews.py \
  --targets_csv <targets_csv> \
  --run_id llama_zs_hausa_v1 \
  --lang hausa \
  --prompt_config configs/gen_zs_config.json
```

### Run evaluation stages
```bash
python3 scripts/eval/run_eval_pipeline.py \
  --stage register \
  --language hausa \
  --run_name qwen_zs_hausa_v1 \
  --gen_csv <generated_csv>
```

Run additional stages (`validate`, `subset`, `dist`, `judge`, `detector`, `master`, `compare`) similarly.

## Key Design Choices
- Stage-based execution with persisted state for debugging and resumability
- CSV-based intermediate artifacts for transparency and auditability
- Language-aware dynamic defaults with alias handling
- Separate judge and detector evaluation tracks merged at master-scoreboard level

## Authorship and Attribution
### Written by me
- Primary orchestration and integration in `scripts/eval/run_eval_pipeline.py`
- Dataset extraction script `languages/extract_dataset.py`
- Generation integration scripts in `scripts/gen/`
- Evaluation wiring and path/default updates in `scripts/eval/`

### Third-party dependencies
- Uses `pandas`, `torch`, and `transformers`

### Notes
- This public sample excludes private credentials, non-public data, and restricted internal assets.
- Any externally authored code should be attributed at file level if included.

## Related Publication (if applicable)
- `MAiDE-up: Multilingual Deception Detection of GPT-generated Hotel Reviews`
- https://arxiv.org/pdf/2404.12938
