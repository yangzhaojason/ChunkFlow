# ChunkFlow

ChunkFlow is the research implementation for **“ChunkFlow: Towards Continuity-Consistent Chunked Policy Learning,”**
accepted by the **IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2026**.

Authors: Zhao Yang, Yinan Shi, Mingyuan Yao, Wenyao Xue, Yawei Jueluo, and
Longjun Liu. See the [project page](https://cytoderm-ai.github.io/chunkflow/)
and the repository [citation metadata](CITATION.cff).

## Lineage and scope

This is a derivative of [Physical Intelligence's openpi](https://github.com/Physical-Intelligence/openpi)
and retains its complete π0/π0-FAST/π0.5 training and inference foundation. The
Python namespace intentionally remains `openpi`, so existing openpi imports and
base workflows continue to work. The ChunkFlow authors are the authors of the
ChunkFlow additions; they are not presented as the upstream π0/π0.5 authors.

The repository contains the source needed to train, evaluate, serve, and extend
ChunkFlow without a separate source checkout. External datasets, model weights,
and CALVIN are not bundled; obtain them from their respective providers and
accept their licenses before use.

## What ChunkFlow adds

| Capability | Repository support |
| --- | --- |
| RTC overlap execution | Deterministic overlap blending during rollout |
| Executed-history conditioning | The next policy call is conditioned on actions actually returned to the environment |
| First-order continuity | Total-variation regularization on predicted action chunks |
| Second-order continuity | Curvature regularization on predicted action chunks |
| Stopped target-relative boundary surrogate | Residual seam alignment for adjacent supervised chunks |
| BC → step-wise AWAC | Two-stage supervised training followed by optional reward-aware fine-tuning |
| AWAC backend | JAX only |
| Critic lifecycle | Training only; serving and actor checkpoints do not require the critic |

Because π0.5 is an implicit flow policy, the AWAC actor term is a
flow-matching surrogate, not an exact log-probability objective. The frozen-actor
reference-consistency term is not an exact KL, and exact entropy is unsupported.
See [the mathematical and dataset contract](docs/chunkflow_awac.md) and the
[paper-to-code map](docs/paper_to_code.md) for the precise implementation scope.

## Requirements

The inherited openpi hardware guidance applies. A supported setup uses an
NVIDIA GPU and Ubuntu 22.04. Approximate single-GPU memory requirements are:

| Mode | Memory | Example |
| --- | ---: | --- |
| Inference | more than 8 GB | RTX 4090 |
| LoRA fine-tuning | more than 22.5 GB | RTX 4090 |
| Full fine-tuning | more than 70 GB | A100 80 GB or H100 |

Multiple GPUs can be selected with `fsdp_devices`; the JAX trainer does not
support multi-node training. Install [uv](https://docs.astral.sh/uv/) before
setting up Python dependencies.

## Installation

```bash
git clone git@github.com:yangzhaojason/ChunkFlow.git
cd ChunkFlow
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

`GIT_LFS_SKIP_SMUDGE=1` avoids downloading LeRobot's large files while resolving
the dependency. A container alternative is documented in [Docker setup](docs/docker.md).

## Training the paper configuration

The canonical configs are `pi05_chunkflow_paper_bc` and
`pi05_chunkflow_paper_awac`. They use chunk length `L=10`, overlap `O=8`, stride
`S=2`, and executed-history length `p=4`. The expected LeRobot records and AWAC
step-transition fields are specified in [the AWAC workflow](docs/chunkflow_awac.md).

Set the dataset and initialization locations before importing the training
configuration:

```bash
export CHUNKFLOW_PAPER_REPO_ID=your-org/chunkflow-lerobot
export CHUNKFLOW_PAPER_DATASET_ROOT=datasets/chunkflow_lerobot
export CHUNKFLOW_PI05_BASE_CHECKPOINT=gs://openpi-assets/checkpoints/pi05_base/params
```

Replace `your-org/chunkflow-lerobot` with the actual LeRobot repository ID. A
local dataset root can be used without publishing the data. Dataset conversion
examples are available for [LIBERO](examples/libero/convert_libero_data_to_lerobot.py),
[ALOHA](examples/aloha_real/convert_aloha_data_to_lerobot.py), and
[DROID](examples/droid/convert_droid_data_to_lerobot.py).

### 1. Compute normalization statistics

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_chunkflow_paper_bc
```

The statistics are written under `assets/pi05_chunkflow_paper_bc`. See
[normalization details](docs/norm_stats.md) when adapting the transforms or
reusing statistics.

### 2. Supervised behavior cloning

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_chunkflow_paper_bc --exp-name chunkflow_bc --overwrite
```

### 3. Step-wise AWAC fine-tuning

The weight initializer points to the BC step's `params` directory. Its sibling
`assets` directory supplies the exact normalization statistics used by that
actor.

```bash
export CHUNKFLOW_SUPERVISED_CHECKPOINT=checkpoints/pi05_chunkflow_paper_bc/chunkflow_bc/29999/params
export CHUNKFLOW_SUPERVISED_ASSETS=checkpoints/pi05_chunkflow_paper_bc/chunkflow_bc/29999/assets
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_chunkflow_paper_awac --exp-name chunkflow_awac --overwrite
```

To resume an AWAC run, restore its complete state rather than initializing a new
one from BC:

```bash
uv run scripts/train.py pi05_chunkflow_paper_awac --exp-name chunkflow_awac --resume
```

## Checkpoint layout

A checkpoint step root contains both `params/` and `assets/`:

```text
checkpoints/pi05_chunkflow_paper_awac/chunkflow_awac/29999/
├── params/
└── assets/
```

Use `.../29999/params` only for the BC-to-AWAC weight initializer. Policy
serving and evaluation `checkpoint-dir` values must point to the step root
`.../29999`, because policy construction loads both directories.

## Inference and evaluation

### Raw actor policy server

The standard openpi WebSocket command below loads the AWAC actor with
`create_trained_policy`. It is a raw actor policy server: it does not perform
overlap blending and does not maintain executed history, so it is not a complete
ChunkFlow RTC runtime. The critic is not loaded for inference.

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_chunkflow_paper_awac \
  --policy.dir=checkpoints/pi05_chunkflow_paper_awac/chunkflow_awac/29999
```

For local stateful execution, wrap that trained actor with the repository's RTC
policy and reset it at every episode boundary:

```python
from openpi.policies import policy_config
from openpi.policies.rtc_policy import RTCConfig, RTCPolicy
from openpi.training import config as _config

config = _config.get_config("pi05_chunkflow_paper_awac")
checkpoint_dir = "checkpoints/pi05_chunkflow_paper_awac/chunkflow_awac/29999"
base = policy_config.create_trained_policy(config, checkpoint_dir)
rtc_config = RTCConfig.from_model_config(
    config.model,
    overlap_size=8,
    replan_interval=2,
    blending_method="linear",
    track_metrics=True,
)
rtc_policy = RTCPolicy(base, rtc_config)


def run_episode(observations):
    rtc_policy.reset()
    for observation in observations:
        yield rtc_policy.infer(observation)["actions"]
```

The ready-made complete RTC benchmark paths are the CALVIN and LIBERO wrappers
below. For remote deployment, keep `RTCPolicy` episode state in the robot control
loop or in a stateful service. A raw actor WebSocket client that does not wrap
the actor with RTC is not equivalent to ChunkFlow RTC. Transport and client
integration details are covered by [remote inference](docs/remote_inference.md).

### CALVIN

Install CALVIN separately, prepare its dataset, and pass absolute or
working-directory-relative locations to the launcher:

```bash
CALVIN_ROOT=third_party/calvin \
CALVIN_DATASET=datasets/calvin/task_D_D \
CHUNKFLOW_CHECKPOINT=checkpoints/pi05_chunkflow_paper_awac/chunkflow_awac/29999 \
CHUNKFLOW_CONFIG=pi05_chunkflow_paper_awac \
CALVIN_OUTPUT_DIR=outputs/calvin/chunkflow \
bash eval_code/eval_calvin_chunkflow.sh \
  --overlap-size 8 \
  --replan-interval 2
```

The launcher also accepts `CALVIN_NUM_SEQUENCES` and `PYTHON_BIN`. It does not
download CALVIN code, assets, or data.

### LIBERO

Install LIBERO in the environment (the evaluator also recognizes a checkout at
`third_party/libero`) and invoke the evaluator directly:

```bash
uv run python eval_code/pi05_rtc_libero_eval.py \
  --config pi05_chunkflow_paper_awac \
  --checkpoint-dir checkpoints/pi05_chunkflow_paper_awac/chunkflow_awac/29999 \
  --task-suite-name libero_10 \
  --output-dir outputs/libero/chunkflow \
  --overlap-size 8 \
  --replan-interval 2
```

Both benchmark commands explicitly match the paper config's `O=8` and `S=2`
(`replan_interval=2`) instead of inheriting the evaluator defaults. Adjust
`--task-suite-name` and trial count for the installed benchmark version. The
upstream-style Docker workflow remains in the [LIBERO example](examples/libero/README.md).

### Action smoothness

LIBERO evaluation writes per-episode action arrays below its output directory.
Compute first/second/third differences, seam metrics, total variation, and
high-frequency energy with:

```bash
uv run python eval_code/eval_action_smoothness.py \
  --pred_dir outputs/libero/chunkflow/data/pred_actions \
  --act_seq_len 10 \
  --stride 10 \
  --output outputs/libero/chunkflow/smoothness.json
```

The smoothness script also accepts `--pred_file` for one action sequence or
`--root_data_dir` for compatible frame-wise ground truth.

## Preserved openpi foundation

ChunkFlow keeps the upstream base model and robot workflows available:

- **π0** flow-based VLA and **π0-FAST** autoregressive VLA with the FAST action
  tokenizer.
- **π0.5** flow-matching training and inference, used as the paper's base actor.
- Upstream checkpoint loading and fine-tuning paths for ALOHA, DROID, and
  LIBERO. Base and expert model weights remain hosted by Physical Intelligence;
  they are not republished in this repository.
- Robot examples for [ALOHA simulation](examples/aloha_sim),
  [ALOHA real](examples/aloha_real), [DROID](examples/droid/README.md),
  [LIBERO](examples/libero/README.md), and [UR5](examples/ur5).
- Local and remote inference through the standard `openpi` policy API,
  [remote inference](docs/remote_inference.md), and the
  [simple client](examples/simple_client/README.md).

### PyTorch boundary

The inherited basic PyTorch support for π0 and π0.5 inference/fine-tuning is
preserved, including [JAX-to-PyTorch conversion](examples/convert_jax_model_to_pytorch.py)
and [`scripts/train_pytorch.py`](scripts/train_pytorch.py). ChunkFlow AWAC is not supported by PyTorch;
use the JAX commands above. RTC actor inference remains a
policy wrapper and does not require an AWAC critic.

## Development and attribution

See [CONTRIBUTING.md](CONTRIBUTING.md) for development checks and issue links.
ChunkFlow is distributed under the [Apache License 2.0](LICENSE). Upstream and
embedded-source attribution is recorded in [NOTICE](NOTICE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md); retain the original file
headers when redistributing modified copies.
