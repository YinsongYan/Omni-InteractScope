<p align="center">
  <img src="assets/readme/omni-interactscope-mark.svg" alt="Omni-InteractScope logo: vision and sound connected by a live interaction loop" width="230">
</p>

<h1 align="center">Omni-InteractScope</h1>

<p align="center"><strong>Benchmarking multimodal interaction in streaming video</strong></p>

<p align="center">
  <a href="https://huggingface.co/datasets/Darden7/Omni-InteractScope"><img alt="Dataset on Hugging Face" src="https://img.shields.io/badge/Dataset-Hugging%20Face-FFD21E?style=for-the-badge&amp;logo=huggingface&amp;logoColor=black"></a>
  <a href="#get-the-code-and-data"><img alt="Get started" src="https://img.shields.io/badge/Quick%20Start-Get%20Started-25AFA8?style=for-the-badge"></a>
  <a href="LEADERBOARD.md"><img alt="Leaderboard" src="https://img.shields.io/badge/Leaderboard-Results-7465D9?style=for-the-badge"></a>
</p>

<p align="center">
  <a href="docs/data.md">Data Format</a> ·
  <a href="docs/models.md">Model Setup</a> ·
  <a href="docs/protocol.md">Evaluation Protocol</a> ·
  <a href="LICENSE">Code License</a>
</p>

Omni-InteractScope evaluates multimodal interaction models across **1,479 interactions, 1,790 reference response targets, and eight task groups (T1–T8)**, measuring both response content and timing. Evaluation code and documentation are here; the [dataset](https://huggingface.co/datasets/Darden7/Omni-InteractScope) is on Hugging Face.

## Repository layout

The code lives at the GitHub repository root. Downloading the Hugging Face dataset into this checkout creates `data/Omni-InteractScope/`, the default path used by the evaluation scripts.

```text
Omni-InteractScope/
  README.md                       # setup, downloads, and evaluation
  LICENSE                         # Apache-2.0 code license
  pyproject.toml                   # package metadata and dependencies
  .env.example                    # credential variable names, with no secrets
  omni_eval/                      # evaluation package and model adapters
  configs/                        # model, task, and evaluation configurations
  requirements/                   # optional model-specific dependencies
  scripts/                        # evaluation and external-video import tools
  tests/                          # regression tests
  docs/                           # data, model, and protocol documentation
  LEADERBOARD.md                   # reference results
  data/                            # downloaded separately from Hugging Face; ignored by Git
    Omni-InteractScope/
      annotations.json             # all 1,479 interaction records
      videos/                      # 207 hosted self-recorded videos
      query_attachments/           # query audio and images
      asr_cache/                   # fixed ASR inputs
      file_manifest.jsonl
      SHA256SUMS
      LICENSE.md                   # CC BY-NC 4.0 dataset terms
```

## What is included

| Component | Availability |
|---|---|
| Evaluation code, configurations, scripts, tests, and documentation | This GitHub repository |
| Annotations, query attachments, ASR caches, and verification manifests | Hugging Face dataset, under `data/Omni-InteractScope/` |
| 207 self-recorded videos | Hugging Face dataset; approximately 27.43 GB |
| 905 videos sourced from Douyin, Xiaohongshu, and Bilibili | Source links and reference metadata only; video files are not hosted |

The hosted dataset occupies approximately **27.46 GB**. The complete reference video collection contains 1,112 videos and occupies approximately 151.3 GB. Full benchmark evaluation requires obtaining the 905 external videos separately; downloading the Hugging Face repository does not include those videos.

## Get the code and data

Use Python 3.10 or later. Linux with `ffmpeg` and `ffprobe` is recommended for evaluation. Sign in with a Hugging Face account that has access to the dataset. The GitHub checkout contains no benchmark data.

```bash
git clone https://github.com/YinsongYan/Omni-InteractScope.git
cd Omni-InteractScope

python3 -m venv .venv-omni
source .venv-omni/bin/activate
python -m pip install --upgrade huggingface_hub
hf auth login

# Download only the hosted dataset into this code checkout.
hf download Darden7/Omni-InteractScope --repo-type dataset \
  --revision main --include 'data/**' --local-dir .
```

## Install the evaluation environment

From the repository root, with the Python environment still activated:

```bash
# Ubuntu / Debian
sudo apt-get install -y ffmpeg
python -m pip install -e '.[media,judge,download]'
```

For local models, install the appropriate PyTorch/CUDA stack and the model-specific dependencies described in the [model setup guide](docs/models.md). Hosted API credentials are documented in [`.env.example`](.env.example); export the required variables in your shell, since the harness does not automatically load `.env` files.

## Prepare and verify the dataset

For each external video, `source_attribution.source` and `source_attribution.original_video_url` in [the hosted `annotations.json`](https://huggingface.co/datasets/Darden7/Omni-InteractScope/blob/main/data/Omni-InteractScope/annotations.json) identify its source. Prepare the clip using `source_start_ms` and `source_end_ms`, retain its audio, and save it at the relative `video` path. Multiple interactions may reference one file. The [video access guide](docs/data.md#video-access) explains link export, clipping, checksums, and importing an existing archive.

The manifests describe the complete reference dataset, including the 905 external videos. Once all reference files are available, run:

```bash
omni-bench verify --dataset data/Omni-InteractScope
omni-bench verify --dataset data/Omni-InteractScope --checksums
```

## Evaluate a model

Install the selected model and configure its credentials first. Content scoring uses `gpt-4o-2024-08-06`; uncached content comparisons make paid API calls. T1 tool-set scoring is deterministic.

```bash
export OPENAI_API_KEY='your-judge-key'
export OPENAI_BASE_URL='https://api.openai.com/v1'

# After installing Qwen3-VL and activating its environment from docs/models.md:
python scripts/evaluate.py --model qwen3-vl-8b \
  --dataset data/Omni-InteractScope --output-dir outputs
```

The script runs inference, scoring, and CSV/Markdown export. Evaluation covers **explicit-request tasks (T1–T7)** and **assistance tasks (T8)**. CLI splits select records from the same `annotations.json`:

| Evaluation setting | CLI split | Tasks | Interactions / targets | Metrics |
|---|---|---|---:|---|
| Explicit-request tasks (T1–T7) | `table2` | T1–T4, T6, T7 | 1,063 / 1,313 | Per-task accuracy and equal-weight Avg. |
| Explicit-request tasks (T1–T7) | `t5` | T5, streaming models only | 115 / 115 | Separately reported temporal-condition accuracy |
| Assistance tasks (T8) | `table3` | T8-A/B/C | 299 / 360 | Overall/subtype recall and Complete T8-C |

All 12 configurations support `table2` and `table3`. LiveCC, AURA, MMDuet2, MOSS-VL, and MiniCPM-o 4.5 also support `t5`. T5 is excluded from Avg. See the [protocol](docs/protocol.md) for task definitions and timing/matching rules.

```bash
# Print commands without loading a model or calling APIs
python scripts/evaluate.py --model qwen3-vl-8b --dry-run
# Run individual stages or T5 alone
python scripts/evaluate.py --model qwen3-vl-8b --stage infer
python scripts/evaluate.py --model qwen3-vl-8b --stage score
python scripts/evaluate.py --model minicpm-o --splits t5
```

Results are written to `outputs/<model>/`: predictions in `table2.jsonl` / `table3.jsonl`, scores in `scores/`, and exports in `tables/`. Keep each prediction file's `.run.json` sidecar. Rerun the same command to resume; use a new `--output-dir` when changing data or configuration. Judge decisions are cached in `.cache/judge.jsonl`.

Use `omni-bench infer`, `omni-bench score`, and `omni-bench tables` directly for custom configurations, GPU sharding, or small smoke runs. See [model commands](docs/models.md).

## Development

- `omni_eval/`: model adapters, media input, protocols, and scoring.
- `configs/` and `requirements/`: model configurations and dependencies.
- `scripts/evaluate.py`: evaluation entry point; `scripts/materialize_public_videos.py`: import existing video files.
- `tests/`: evaluation tests. Install `pip install -e '.[dev,media,judge]'`, then run `pytest`.

## Code license

The evaluation code in this repository is licensed under the [Apache License 2.0](LICENSE). The dataset, including annotations, audio, video, and images, is governed separately by the [dataset license on Hugging Face](https://huggingface.co/datasets/Darden7/Omni-InteractScope/blob/main/data/Omni-InteractScope/LICENSE.md); the Apache-2.0 code license does not apply to the dataset. Copyright and related rights in third-party materials remain with their respective rights holders.

## Dataset license and use restrictions

This dataset is released for non-commercial research use under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

This dataset is intended solely for non-commercial academic research and model evaluation. It must not be used in commercial products, services, or other profit-making activities. For videos sourced from Douyin, Xiaohongshu, and Bilibili, this repository provides only index references and textual annotations; it does not include, copy, or redistribute the original videos or their audio tracks. Self-recorded videos, query audio, and image attachments provided in the repository are also subject to this dataset license.

Copyright and related rights in the original images, audio, and videos remain with their respective creators and rights holders. The CC BY-NC 4.0 license applies to rights held by the dataset contributors and does not relicense third-party materials. When accessing or using referenced or provided materials, users are responsible for complying with the source platforms' terms of service, the applicable licenses and permissions for the source materials, and applicable laws and regulations concerning personal information protection and portrait rights.
