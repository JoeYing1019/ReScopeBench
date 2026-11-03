<h1 align="center">ReScopeBench: Towards Robust Long-Lived Agents</h1>
<p align="center"><b>Benchmarking Experience Lifecycles in Evolving Environments</b></p>

<div align="center">

[![Homepage](https://img.shields.io/badge/Homepage-ReScopeBench-225F96)](https://rescopebench.github.io/)
[![Dataset](https://img.shields.io/badge/Dataset-Hugging%20Face-3276B3)](https://huggingface.co/datasets/JoeYing/ReScopeBench)

</div>

<p align="center">
  <a href="#overview">Overview</a> |
  <a href="#method">Method</a> |
  <a href="#main-results">Main Results</a> |
  <a href="#installation">Installation</a> |
  <a href="#evaluation">Evaluation</a> |
  <a href="#metrics">Metrics</a>
</p>

## Overview

**ReScopeBench** evaluates how agents form, reuse, and revise experience in
evolving environments. It contains **18 stateful environments, 352 tasks, and
26 drift events**, covering changes to structural knowledge and operational rules.

Each model completes two independent task streams per environment:
**w exp** retains naturally produced memory, skills, and permitted history;
**w/o exp** clears explicit experience between tasks. Both settings start from
the same initial environment and maintain their own business state.

This repository provides the evaluation harness and metric computation.
Executable cases are hosted on [Hugging Face](https://huggingface.co/datasets/JoeYing/ReScopeBench).
The agent harness builds on [Hermes Agent](https://github.com/NousResearch/hermes-agent);
its [MIT license](rescopebench/vendor/HERMES_LICENSE) accompanies the bundled runtime.

## Method

ReScopeBench uses experience-first construction: tasks share discoverable
structural and rule knowledge, while controlled changes selectively invalidate
parts of that knowledge.

- **Structural experience:** how resources are organized, related, and accessed.
- **Rule experience:** conditions, constraints, and consequences that guide actions.

<p align="center">
  <a href="fig/method.png"><img src="fig/method.png" width="95%" alt="ReScopeBench construction and experience lifecycle" /></a>
</p>

## Main Results

The paper evaluates 20 models across all 18 environments with and without
cross-task experience.

<p align="center">
  <a href="fig/main_results.png"><img src="fig/main_results.png" width="95%" alt="ReScopeBench main results" /></a>
</p>

## Installation

Use **Python 3.10** and Docker with **linux/arm64** support. The tested host
platform is macOS/arm64. Allow at least **25 GiB of free disk space** for setup
and execution.

Run these commands from the repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r rescopebench/requirements.txt huggingface_hub
python scripts/download_cases.py
python scripts/restore_runtime.py
bash hermes_image/build_local.sh
export RESCOPE_SANDBOX_IMAGE=rescopebench-harness:local
```

The installer downloads the pinned cases and checks their SHA-256 hashes.
Each environment has one `driver.py` and one `formal-binding.json`.
The runtime image is restored from the bundled parts and built locally.

## Evaluation

### Configure a Model

Set your service endpoint and served model ID. Supply `OPENAI_API_KEY` through
a secret manager or a hidden terminal prompt.

```bash
export RESCOPE_SANDBOX=docker
export RESCOPE_PROVIDER_PROTOCOL=openai
export RESCOPE_MODEL_API=openai-chat
export PROVIDER_API=chat_completions
export OPENAI_BASE_URL=https://your-provider.example/v1
export MODEL=your-served-model-id

printf 'Model API key: '
read -r -s OPENAI_API_KEY
export OPENAI_API_KEY
printf '\n'
```

For the Responses API, set `RESCOPE_MODEL_API=openai-responses` and
`PROVIDER_API=responses`. Native Anthropic and Gemini adapters use
`anthropic-messages` and `gemini-generate-content`, respectively, with
`PROVIDER_API=chat_completions`.

### Run Both Settings

The following example evaluates Robot. Replace `robot` with another case
directory name to evaluate a different environment.

```bash
OUT="$PWD/results/robot-model-a"

python -m rescope_harness eval prepare \
  --case robot --output "$OUT" --model "$MODEL" \
  --provider-api "$PROVIDER_API" --tool-execution batched --execution-mode formal
python -m rescope_harness eval verify --case robot --output "$OUT"

for ARM in native reset; do
  python -m rescope_harness eval run-arm \
    --case robot --output "$OUT" --arm "$ARM" \
    --no-board-register --retry-infrastructure
done
```

`native` is **w exp**; `reset` is **w/o exp**. Run both complete streams for
each model and environment. For an initial smoke run, set `RESCOPE_LIMIT=1`;
unset it and repeat `run-arm` to continue the same stream.

`batched` lets the model request multiple tool calls in one turn. Calls execute
in model order, with each operation charged separately under the environment's
cost contract. This is independent of concurrent stream execution.

For multiple prepared runs, populate a registry using
[the example](config/registry.example.json), then launch:

```bash
python -m rescope_harness batch --registry registry.json \
  --output "$PWD/controllers/batch-a" --workers 2 --retry-infrastructure
```

`workers` counts concurrent arms. Use a new output directory when changing the
model or runtime configuration. Completion is recorded in each arm's
`summary.json`.

## Metrics

The implementation reports seven scores on a 0-100 scale, with equal weighting
across environments:

- **Performance:** `B` (w/o exp), `W` (w exp), and `D` (drift-onset performance).
- **Experience:** `S` (structural coverage), `R` (rule coverage), `U` (immediate updating), and `H` (recovery timeliness).

The paper labels these **BP / EP / DP / SC / RC / UC / RT**. `AVG` is their
arithmetic mean. Experience scores measure coverage of reference knowledge in
saved memory and skills, accepting semantically equivalent expressions.

Business scores can be computed without a judge:

```bash
python -m rescope_metrics performance \
  --run "$OUT" --output "$PWD/scores/robot-business.json"
```

To judge saved experience and produce all seven metrics, configure an accessible
judge and securely supply `RESCOPE_JUDGE_API_KEY`:

```bash
python -m rescope_metrics prepare --run "$OUT" \
  --output "$PWD/scores/robot-model-a" \
  --judge-model your-judge-model \
  --judge-base-url https://your-judge.example/v1 --judge-api responses
python -m rescope_metrics judge --output "$PWD/scores/robot-model-a" --workers 16
python -m rescope_metrics score --output "$PWD/scores/robot-model-a"
python -m rescope_metrics report \
  --inputs "$PWD"/scores/*/metrics.json --output "$PWD/reports/main"
```

Experience evaluation uses frozen claim rubrics and one request per experience.
Use the same judge configuration across models. Only `judge` calls a model;
the other metric commands are local and require no Docker container.
The report includes CSV results and a self-contained HTML visualization.

## Acknowledgements

This research was financially supported by the Open Research Fund from Guangdong Laboratory of Artificial Intelligence and Digital Economy (SZ), under Grant No. GML-KF-26-06.
