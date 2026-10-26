# Model environment setup and evaluation commands

Run all commands from the repository root. Install Linux `ffmpeg` / `ffprobe` first
(see the [README](../README.md#first-time-environment-setup)). Examples use Python 3.10 or newer and write
predictions, scores and tables to `outputs/<model>/`.

Each evaluation command runs all applicable tasks: T1–T4/T6/T7 (`table2`), T8 (`table3`),
and additionally T5 (`t5`) for streaming models. Add `--stage infer` to generate predictions
without scoring, or `--stage score` to score existing predictions and export tables.
Add `--dataset /path/to/dataset` if your data is outside `data/Omni-InteractScope`.

## Environment organization

These instructions assume you are using the repository for the first time. Create the
environment for your selected model, install its dependencies, then run evaluation in that
same activated environment. Each setup block below is self-contained. Do not install every
requirements file into one environment.

| Environment | Models / purpose |
|---|---|
| `.venv-api` | Gemini-3.1-Pro, Gemini-3.5-Flash, Qwen3.5-Omni-Plus and content scoring |
| `.venv-qwen` | Qwen3-VL, Qwen2.5-Omni and Qwen3-Omni (shared compatible requirements) |
| `.venv-internvl` | InternVL3.5 |
| `.venv-livecc` | LiveCC |
| `.venv-mmduet2` | MMDuet2 |
| `.venv-moss` | MOSS-VL-Realtime |
| `.venv-minicpm` | MiniCPM-o 4.5 |
| `.venv-aura-server` | AURA vLLM server |
| `.venv-aura-client` | AURA evaluation client and scoring |

Environments live under the checkout and are ignored by Git. Setup is needed only once per
environment. In a new terminal, activate it again, for example `source .venv-livecc/bin/activate`,
then run the evaluation command. Switching environments changes which Python and model
libraries the evaluator uses. No environment from our evaluation machines is required.

## Model roster

| `--model` | Checkpoint / API model | Input | Environment | Tasks |
|---|---|---|---|---|
| `gemini-3.1-pro` | `gemini-3.1-pro-preview` | V, A | [Hosted API](#hosted-api-models) | T1–T4, T6–T8 |
| `gemini-3.5-flash` | `gemini-3.5-flash` | V, A | [Hosted API](#hosted-api-models) | T1–T4, T6–T8 |
| `qwen3.5-omni-plus` | `qwen3.5-omni-plus` | V, A | [Hosted API](#hosted-api-models) | T1–T4, T6–T8 |
| `internvl3.5-8b` | `OpenGVLab/InternVL3_5-8B` | V+ASR | [InternVL](#internvl35-8b) | T1–T4, T6–T8 |
| `qwen3-vl-8b` | `Qwen/Qwen3-VL-8B-Instruct` | V+ASR | [Qwen](#qwen-local-models) | T1–T4, T6–T8 |
| `qwen2.5-omni-7b` | `Qwen/Qwen2.5-Omni-7B` | V, A | [Qwen](#qwen-local-models) | T1–T4, T6–T8 |
| `qwen3-omni-30b-a3b` | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | V, A | [Qwen](#qwen-local-models) | T1–T4, T6–T8 |
| `livecc-7b` | `chenjoya/LiveCC-7B-Instruct` | V | [LiveCC](#livecc) | T1–T8 |
| `aura` | `aurateam/AURA` | V | [AURA server + client](#aura) | T1–T8 |
| `mmduet2` | `wangyueqian/MMDuet2` | V | [MMDuet2](#mmduet2) | T1–T8 |
| `moss-vl-realtime` | `OpenMOSS-Team/MOSS-VL-Realtime` | V | [MOSS-VL](#moss-vl-realtime) | T1–T8 |
| `minicpm-o` | `openbmb/MiniCPM-o-4_5` | V, A | [MiniCPM-o 4.5](#minicpm-o-45) | T1–T8 |

V is video; A is native audio; V+ASR supplies scene transcripts. The dataset includes the
required ASR caches. Historical spoken requests use ASR for vision-only models; T8 uses
annotated query text. Input settings and generation parameters are selected automatically
from `configs/models/<model>.yaml`.

## Content judge (all models)

Before running scoring, export the judge credentials in the active terminal:

```bash
export OPENAI_API_KEY='your-judge-key'
export OPENAI_BASE_URL='https://api.openai.com/v1'
```

The binary content judge is `gpt-4o-2024-08-06`; uncached comparisons incur API charges.
The judge key is separate from inference keys below. `.env.example` lists the variables;
the evaluator does not automatically load `.env` files.

## Hosted API models

No local model weights or CUDA are needed. Create the API/scoring environment once:

```bash
python3 -m venv .venv-api
source .venv-api/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[media,judge,download]'
```

Use your own official provider keys. The supplied model configurations point directly to
Google's Gemini API and Alibaba Cloud's DashScope API. No shared service credentials are
included. The content judge separately uses your `OPENAI_API_KEY`.

### Gemini-3.1-Pro and Gemini-3.5-Flash

```bash
export GEMINI_API_KEY='YOUR_GOOGLE_AI_STUDIO_KEY'

python scripts/evaluate.py --model gemini-3.1-pro
python scripts/evaluate.py --model gemini-3.5-flash
```

The endpoint is `https://generativelanguage.googleapis.com/v1beta/models`.
Get your key from [Google AI Studio](https://aistudio.google.com/apikey).

### Qwen3.5-Omni-Plus

```bash
export QWEN_OMNI_API_KEY='YOUR_DASHSCOPE_KEY'

python scripts/evaluate.py --model qwen3.5-omni-plus
```

The endpoint is `https://dashscope.aliyuncs.com/compatible-mode/v1`.
Create your key in [Alibaba Cloud Model Studio](https://bailian.console.aliyun.com/)
for the region served by this endpoint. The key must have access to the configured model.

Each command runs inference, scoring and table export for T1–T4/T6–T8; T5 is N/A for these
API models. To generate predictions first, append `--stage infer`; later rerun with
`--stage score` and your judge key. Add `--workers 4` for concurrent API inference if your
rate limit permits it. Provider model availability depends on your account and region.

## Local GPU environment

Keep a separate environment for each family: the requirements use different Transformers
versions. The commands below use a **CUDA 12.8 / PyTorch 2.8 installation example**, not a
claim that every hardware configuration has been tested. Use matching PyTorch, torchvision
and torchaudio wheels for your GPU/driver. MOSS-VL's `torchcodec==0.7.0` uses PyTorch 2.8;
AURA has its own pinned server environment below.

Qwen, InternVL, LiveCC, MMDuet2 and MOSS-VL use FlashAttention 2. Building it requires a
compatible CUDA toolkit (`nvcc`), a C++ compiler and the build dependencies shown below.
MiniCPM-o uses SDPA and does not need FlashAttention. Allow sufficient GPU memory for the
selected weights and context; Qwen3-Omni loads a 30B-parameter model even though 3B are active.

### Qwen local models

One Qwen environment supports the three local Qwen configurations. Run only the evaluation
commands for the models you want:

```bash
python3 -m venv .venv-qwen
source .venv-qwen/bin/activate
python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[media,judge]' -r requirements/qwen.txt
python -m pip install 'flash-attn>=2.8,<3' --no-build-isolation

CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --model qwen3-vl-8b
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --model qwen2.5-omni-7b
CUDA_VISIBLE_DEVICES=0,1 python scripts/evaluate.py --model qwen3-omni-30b-a3b
```

These adapters use `device_map=auto` and can place weights across the visible GPUs.
Requirements: Transformers `>=4.57,<5`, Accelerate, Qwen VL/Omni utilities and audio libraries.

### InternVL3.5-8B

```bash
python3 -m venv .venv-internvl
source .venv-internvl/bin/activate
python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[media,judge]' -r requirements/internvl.txt
python -m pip install 'flash-attn>=2.8,<3' --no-build-isolation

CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --model internvl3.5-8b
```

Requirements: Transformers `>=4.52.1,<5`, Accelerate, timm and einops.

### LiveCC

```bash
python3 -m venv .venv-livecc
source .venv-livecc/bin/activate
python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[media,judge]' -r requirements/livecc.txt
python -m pip install 'flash-attn>=2.8,<3' --no-build-isolation

CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --model livecc-7b
```

Requirements include Transformers `4.51.0`, `qwen-vl-utils==0.0.11`, livecc-utils and
`liger-kernel==0.5.5`. The adapter retains streaming state and uses a 28,000-token context
reset limit. The command includes T5.

### AURA

Start the vLLM server in a separate environment and terminal:

```bash
python3 -m venv .venv-aura-server
source .venv-aura-server/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/aura.txt

CUDA_VISIBLE_DEVICES=0 vllm serve aurateam/AURA \
  --revision 88ee550629fcb4d84428cdbfb346d07f01ea6e03 \
  --served-model-name AURA --host 127.0.0.1 --port 8028 \
  --tensor-parallel-size 1 --max-model-len 32768 --max-num-seqs 1 \
  --gpu-memory-utilization 0.30 --enforce-eager --no-enable-log-requests \
  --limit-mm-per-prompt '{"video":30,"image":1}' \
  --media-io-kwargs '{"video":{"num_frames":-1}}' \
  --mm-processor-kwargs '{"fps":1.0,"size":{"shortest_edge":4096,"longest_edge":151200}}'
```

The server pins vLLM `0.17.1`, PyTorch `2.10.0`, torchvision `0.25.0`, torchaudio `2.10.0`,
Transformers `4.57.6`, OpenAI `2.24.0` and PyAV `17.0.0`. Adjust GPU memory allocation to fit
your server while retaining the model/context settings.

In a second terminal on the same machine, install the evaluation client and set the judge key:

```bash
python3 -m venv .venv-aura-client
source .venv-aura-client/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[media,judge]'
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
export OPENAI_API_KEY='your-judge-key'

python scripts/evaluate.py --model aura
```

The client uses CPU PyTorch for the harness's seed setup; model inference runs on the server.
The endpoint defaults to `http://127.0.0.1:8028/v1` in `configs/models/aura.yaml` and must be
local. The adapter sends causal one-second video segments, retaining up to 30, plus query
context and supported image attachments. It uses silent placeholders for prior assistant
turns. No raw scene audio is sent. The evaluation command includes T5.

### MMDuet2

```bash
python3 -m venv .venv-mmduet2
source .venv-mmduet2/bin/activate
python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[media,judge]' -r requirements/mmduet2.txt
python -m pip install 'flash-attn>=2.8,<3' --no-build-isolation

CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --model mmduet2
```

Requirements: Transformers `4.49.0`, Accelerate and qwen-vl-utils. The configuration uses
second-stage content generation and a 100,000-token context reset limit. The command includes T5.

### MOSS-VL-Realtime

```bash
python3 -m venv .venv-moss
source .venv-moss/bin/activate
python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[media,judge]' -r requirements/moss.txt
python -m pip install 'flash-attn>=2.8,<3' --no-build-isolation

CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --model moss-vl-realtime
```

Requirements include Transformers `4.57.1`, TorchCodec `0.7.0` and einops. The adapter retains
its native session generation and 12-token-per-second pacing. The command includes T5.

### MiniCPM-o 4.5

```bash
python3 -m venv .venv-minicpm
source .venv-minicpm/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[media,judge]' -r requirements/minicpm.txt

CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --model minicpm-o
```

Requirements: Transformers `4.51.0`, `minicpmo-utils[all]>=1.0.2`, librosa and moviepy.
The adapter uses the native duplex interface with a 30-unit window and up to 20 spoken
tokens per chunk. The command includes T5.

## Multiple GPUs

Local inference uses one worker per process. To distribute interactions across two GPUs,
start these commands in separate terminals using the same model environment:

```bash
CUDA_VISIBLE_DEVICES=0 omni-bench infer --model minicpm-o --split table2 \
  --shard 0/2 --output outputs/minicpm-o/table2-shard0.jsonl
CUDA_VISIBLE_DEVICES=1 omni-bench infer --model minicpm-o --split table2 \
  --shard 1/2 --output outputs/minicpm-o/table2-shard1.jsonl
```

After both finish, score them together. Keep each `.run.json` sidecar with its predictions:

```bash
omni-bench score --split table2 \
  --predictions outputs/minicpm-o/table2-shard0.jsonl outputs/minicpm-o/table2-shard1.jsonl \
  --output outputs/minicpm-o/scores/table2.json
omni-bench tables --scores outputs/minicpm-o/scores/table2.json \
  --output-dir outputs/minicpm-o/tables
```

Repeat with `--split table3` or `--split t5` and matching output filenames for the other tasks.
Pass `--dataset` on both inference and scoring if you use a custom dataset location.

## Model configuration and custom adapters

To use a local checkpoint, copy its YAML, change `model_args.model_path`, and pass the copy
to `omni-bench infer --model MODEL --model-config /path/to/model.yaml ...`.
The release configs retain recorded revision pins for LiveCC, AURA, MOSS-VL and MiniCPM-o.
Local prompted models use greedy generation; API models use temperature zero. Other media,
context and generation settings are documented in the YAML configurations.

A custom OpenAI-compatible image/chat server can use the example configuration:

```bash
export BENCH_BASE_URL='http://127.0.0.1:30000/v1'
export BENCH_MODEL='your-served-model'
export BENCH_API_KEY='your-server-key'
omni-bench infer --model openai-compatible --split table2 \
  --output outputs/custom/table2.jsonl
```

For a local server without authentication, use a nonempty placeholder key. The example uses
video frames and query text; the server must accept multiple `image_url` parts. Use the direct
`score` and `tables` commands from the README for its outputs. This custom configuration is
not one of the 12 Leaderboard entries.

For a new Python adapter, implement `ProbeModel.probe(ProbeContext)` for independent calls,
or `StreamingModel.begin(SessionSpec)`, `observe(frame, time_sec, audio_chunk)` and `end()`
for stateful evaluation. Register it in `omni_eval/models/registry.py` and add a model YAML.
Declare input capabilities accurately and propagate generation errors instead of treating
them as abstentions. See [protocol details](protocol.md).
