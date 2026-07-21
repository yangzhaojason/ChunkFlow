# ChunkFlow step-wise AWAC

This repository trains ChunkFlow in two stages: supervised Pi0.5 behavior cloning and optional step-wise AWAC fine-tuning. AWAC training is supported only by the JAX trainer. The critic and every other AWAC-only state are training-time components; policy sampling, serving, and exported actor parameters do not depend on them.

## Dataset contract

The canonical LeRobot configuration reads one record per environment frame. Each record must provide:

- `actions`: a finite real floating behavior action at the current frame, either `[A_raw]` or `[T, A_raw]`. When a sequence is present, its first row is the current executed action; in the canonical dataset `T=L=10`, and the sequence also supplies the supervised action chunk.
- `rewards`: one finite real scalar with raw shape `[]` or `[1]`, containing the immediate reward for that frame.
- `discounts`: one finite real scalar with raw shape `[]` or `[1]`, in `[0, 1]`. Despite the compatibility name, this is a continuation multiplier: `1` for an ordinary continuing transition and `0` when no next frame exists. It must not include gamma because training applies gamma exactly once.
- `episode_index` and `frame_index`: scalar integer identity used to resolve temporal neighbors without relying on storage order.
- `image` and `wrist_image`: one RGB frame each, accepted as `[H, W, 3]` or LeRobot-style `[3, H, W]`; `state`: one low-dimensional vector `[S_raw]`; and task metadata used to construct one prompt string.

The paper configs intentionally set `awac_executed_action_key="actions"`. This mapping is valid only when the recorded `actions` are the post-blending actions actually sent to the environment. A planned action, an unblended chunk target, or another imitation label is not a valid substitute. If a dataset stores executed actions under another field, configure that field explicitly instead of relying on this canonical mapping.

After delta conversion, normalization, and model-width padding, one transition batch contains `executed_action [B, 32]`, `reward [B]`, `continuation [B]`, current and next model-ready observations with the same batch size, and `episode_id [B]`/`frame_index [B]`. Terminal records remain in the stream: they use continuation zero and a shape-valid next-observation stand-in, so the bootstrapped value contributes nothing.

## Two independent training streams

Each AWAC optimizer step zips two independently seeded streams:

1. The supervised stream emits adjacent, episode-aligned chunk pairs. It retains full-chunk flow behavior cloning, executed-action history conditioning, first- and second-order continuity losses, and stopped boundary consistency.
2. The transition stream emits every episode frame as `(s_t, a_t, r_t, c_t, s_{t+1})`, including terminal frames and frames that cannot start a complete supervised chunk. It supplies the critic and advantage-weighted first-action update.

This separation avoids duplicating high-resolution next observations in every paired chunk and avoids changing the two sampling distributions into a chunk-level semi-Markov objective.

## Paper equations and flow-policy surrogates

For Equation (12), the Q target and expectile V objective are implemented as

```text
y_t = r_t + gamma * c_t * target_V(s_{t+1})
L_Q = mean((Q(s_t, a_t) - stop_gradient(y_t))^2)
delta_t = stop_gradient(Q(s_t, a_t)) - V(s_t)
L_V = mean(abs(tau_e - 1[delta_t < 0]) * delta_t^2)
```

`discounts` supplies `c_t`, not `gamma * c_t`. The canonical values are `gamma=0.99` and expectile `tau_e=0.7`.

Equation (13) is implemented with a stopped, positive-advantage weight:

```text
A_t = stop_gradient(Q(s_t, a_t) - V(s_t))
w_t = clip(exp(max(0, A_t) / tau), 1, wmax)
```

The canonical values are `tau=0.05` and `wmax=20`.

Pi0.5 is an implicit flow policy and does not expose a tractable normalized `log pi(a|s)`. The actor therefore uses weighted first-action flow matching as the explicit Equation (14) surrogate. The executed action occupies the first clean-action position, future positions are shape-only dummy values excluded from the loss, and the causal action mask prevents the first output from depending on that dummy tail. This is a flow-matching surrogate, not an exact action log-likelihood.

For the same reason, `kl_beta * reference_consistency` is the explicit Equation (16) KL surrogate. A frozen copy of the supervised actor receives the same observation, history, noisy flow input, time, and dummy tail as the current actor, and training penalizes their squared flow-velocity difference. The canonical flow-field prior coefficient is `kl_beta=2e-4`; neither code nor metrics describe it as an exact KL.

The target V parameters follow an EMA of the updated online V parameters. This is a numerical-stability extension to Equation (12), not a claim about an additional paper equation. Exact flow entropy is unsupported; any nonzero `entropy_lambda` is rejected rather than silently ignored.

The first-order total-variation/continuity and second-order curvature terms directly implement Equation (15). boundary_consistency_loss uses stopped, target-relative residual seam alignment and is explicitly an implementation surrogate/extension. When paired overlap targets share the same coordinate frame, as in canonical LIBERO, it reduces to the paper's raw seam term; for state-relative action transforms, the residual form avoids penalizing coordinate-origin shifts. The full supervised flow/BC loss remains alongside this structural regularization as the imitation anchor; it is not presented here as mathematically part of Equation (15). The reported actor objective also adds the weighted first-action surrogate and reference consistency. Q and V have a separate optimizer objective. Their features are stop-gradient actor features, and the critic is omitted from inference.

## Canonical paper configuration

Both named configs use Pi0.5 with action dimension 32, chunk length `L=10`, overlap `O=8`, stride `S=2`, and executed-history length `p=4`. First-order and second-order continuity weights are both `0.005`, boundary weight is `0.03`, actor EMA decay is `0.999`, batch size is 256, and training runs for 30,000 steps with the same AdamW and cosine schedule as `pi05_libero`.

The paper does not report reproducible numerical values for history noise, history dropout, or EMA scheduled-sampling corruption. The canonical configs therefore keep `history_noise_std`, `history_dropout_probability`, and `history_schedule_max_alpha` at zero. These controls remain configurable for explicit ablations; the repository does not invent nonzero defaults.

Import-time environment controls are:

```text
CHUNKFLOW_PAPER_REPO_ID=chunkflow/libero
CHUNKFLOW_PAPER_DATASET_ROOT=datasets/chunkflow_lerobot
CHUNKFLOW_SUPERVISED_CHECKPOINT=checkpoints/pi05_chunkflow_paper_bc/chunkflow_bc/29999/params
CHUNKFLOW_SUPERVISED_ASSETS=checkpoints/pi05_chunkflow_paper_bc/chunkflow_bc/29999/assets
```

`CHUNKFLOW_PI05_BASE_CHECKPOINT` selects the Pi0.5 initialization for supervised training. `CHUNKFLOW_SUPERVISED_CHECKPOINT` is a weight load into a new AWAC run: it initializes the actor, then creates fresh critic/target state and freezes the loaded actor as the reference. `CHUNKFLOW_SUPERVISED_ASSETS` selects that supervised run's normalization assets. By default it is derived as the checkpoint's sibling `assets` directory; set it explicitly when weights and assets do not share the canonical checkpoint layout. This differs from resuming an existing AWAC checkpoint, which restores all AWAC training state.

## JAX training commands

First compute normalization statistics under the BC config's local assets directory, `assets/pi05_chunkflow_paper_bc`:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_chunkflow_paper_bc
```

Then run supervised behavior cloning:

```bash
uv run scripts/train.py pi05_chunkflow_paper_bc --exp-name chunkflow_bc --overwrite
```

AWAC fine-tuning initialized from the supervised actor:

```bash
CHUNKFLOW_SUPERVISED_CHECKPOINT=checkpoints/pi05_chunkflow_paper_bc/chunkflow_bc/29999/params \
  uv run scripts/train.py pi05_chunkflow_paper_awac --exp-name chunkflow_awac --overwrite
```

Checkpoint saving copies the BC normalization statistics into each checkpoint step's sibling `assets` directory. The canonical AWAC config reuses the supervised checkpoint's copy instead of reading or creating `assets/pi05_chunkflow_paper_awac`. If the assets were copied elsewhere, set `CHUNKFLOW_SUPERVISED_ASSETS` explicitly alongside `CHUNKFLOW_SUPERVISED_CHECKPOINT`.

Resume a complete AWAC run:

```bash
uv run scripts/train.py pi05_chunkflow_paper_awac --exp-name chunkflow_awac --resume
```

All three training commands launch the JAX trainer. `scripts/train_pytorch.py` rejects AWAC before constructing a dataset or loader.
