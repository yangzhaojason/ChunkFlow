# ChunkFlow paper-to-code map

This map connects the method equations in **“ChunkFlow: Towards
Continuity-Consistent Chunked Policy Learning”** to the implementation and its
regression tests. Equation numbers refer to the IROS 2026 paper. For tensor
shapes, dataset fields, coefficients, and runnable training commands, see the
[step-wise AWAC contract](chunkflow_awac.md).

## Supervised policy and execution

### Eq. (3): deterministic overlap execution

- Implementation: [`src/openpi/policies/rtc_policy.py`](../src/openpi/policies/rtc_policy.py)
  aligns the previous chunk tail with the new chunk head, applies linear or
  cosine RTC blending, returns one environment action, and records that
  post-blending action as executed history.
- Tests: [`src/openpi/policies/rtc_policy_test.py`](../src/openpi/policies/rtc_policy_test.py)
  checks stride alignment, endpoint weighting, action-space projection, and the
  requirement that history contain only actions returned to the environment.

### Eq. (5): seam consistency

- Implementation: [`src/openpi/models/chunkflow_losses.py`](../src/openpi/models/chunkflow_losses.py)
  implements `boundary_consistency_loss`, aligns the previous tail with the
  current head, and stops gradients through the previous prediction.
- Tests: [`src/openpi/models/chunkflow_losses_test.py`](../src/openpi/models/chunkflow_losses_test.py)
  checks alignment, stop-gradient behavior, overlap bounds, and target-relative
  residual cancellation.
- Scope: the paper writes the raw seam penalty. The training path can compare
  prediction residuals relative to paired targets; that coordinate-safe form is
  the stopped target-relative residual extension described below.

### Eq. (7): executed-history-conditioned policy

- Model path: [`src/openpi/models/pi0.py`](../src/openpi/models/pi0.py) validates,
  embeds, masks, and attends to `action_history` alongside the π0.5 action
  suffix. Its behavior is covered by
  [`src/openpi/models/pi0_test.py`](../src/openpi/models/pi0_test.py).
- History utilities: [`src/openpi/models/chunkflow_history.py`](../src/openpi/models/chunkflow_history.py)
  validates padded histories; paired supervised and step-transition assembly is
  implemented in
  [`src/openpi/training/chunkflow_batch.py`](../src/openpi/training/chunkflow_batch.py).
- Tests: [`src/openpi/models/chunkflow_history_test.py`](../src/openpi/models/chunkflow_history_test.py)
  and [`src/openpi/training/chunkflow_batch_test.py`](../src/openpi/training/chunkflow_batch_test.py)
  cover masks, episode boundaries, executed-action sources, and transform
  consistency.

### Eq. (8): first- and second-order continuity

- Implementation: `continuity_penalties` in
  [`src/openpi/models/chunkflow_losses.py`](../src/openpi/models/chunkflow_losses.py)
  computes the first-order L1 and second-order squared penalties per chunk.
- Tests: [`src/openpi/models/chunkflow_losses_test.py`](../src/openpi/models/chunkflow_losses_test.py)
  checks known values, short horizons, and a zero subgradient at an exact
  first-order match.

### Eq. (9): history corruption and scheduled sampling

- Implementation: `scheduled_sampling_alpha` and `corrupt_history` in
  [`src/openpi/models/chunkflow_history.py`](../src/openpi/models/chunkflow_history.py)
  apply Gaussian noise, per-position dropout, and stopped interpolation toward
  covered model predictions while preserving padding zeros.
- Integration: [`src/openpi/models/pi0.py`](../src/openpi/models/pi0.py) constructs
  coordinate-aligned predicted history during paired supervised training.
- Tests: [`src/openpi/models/chunkflow_history_test.py`](../src/openpi/models/chunkflow_history_test.py)
  covers scheduling, deterministic random splitting, padding, gradients, and
  JIT behavior. The paper configs leave all three corruption controls at zero
  because the paper does not publish reproducible nonzero values.

### Eq. (10): supervised objective

- Composition: `combine_supervised_losses` in
  [`src/openpi/models/chunkflow_objectives.py`](../src/openpi/models/chunkflow_objectives.py)
  adds full-chunk flow behavior cloning, first-order continuity, second-order
  continuity, and the paired boundary term.
- Training integration: [`src/openpi/models/pi0.py`](../src/openpi/models/pi0.py)
  computes paired endpoints and metrics; the compatibility exports in
  [`src/openpi/training/chunkflow_train.py`](../src/openpi/training/chunkflow_train.py)
  are used by both training modes.
- Tests: [`src/openpi/models/pi0_test.py`](../src/openpi/models/pi0_test.py) and
  [`src/openpi/training/chunkflow_train_test.py`](../src/openpi/training/chunkflow_train_test.py)
  cover paired-flow composition, target alignment, and batch compatibility.

## Step-wise AWAC fine-tuning

### Eq. (12): Q TD regression and expectile V fitting

- Numerical objectives: `td_targets` and `expectile_value_loss` in
  [`src/openpi/models/chunkflow_awac.py`](../src/openpi/models/chunkflow_awac.py).
- Critic: [`src/openpi/models/chunkflow_critic.py`](../src/openpi/models/chunkflow_critic.py)
  defines separate Q and V heads over stop-gradient actor features; it is
  training-only state.
- Integration: [`src/openpi/training/chunkflow_awac_train.py`](../src/openpi/training/chunkflow_awac_train.py)
  applies gamma exactly once, masks terminal bootstrap with continuation, and
  optimizes the critic separately from the actor.
- Tests: [`src/openpi/models/chunkflow_awac_test.py`](../src/openpi/models/chunkflow_awac_test.py),
  [`src/openpi/models/chunkflow_critic_test.py`](../src/openpi/models/chunkflow_critic_test.py),
  and [`src/openpi/training/chunkflow_awac_train_test.py`](../src/openpi/training/chunkflow_awac_train_test.py).

The target-V parameters follow an EMA of updated online V parameters. This
target-V EMA is a stability extension to Equation 12, not another paper loss.

### Eq. (13): clipped advantage weights

- Implementation: `clipped_advantage_weights` in
  [`src/openpi/models/chunkflow_awac.py`](../src/openpi/models/chunkflow_awac.py)
  stops the advantage, retains its positive part, exponentiates by temperature,
  and clips to `[1, wmax]` with finite low-precision handling.
- Tests: [`src/openpi/models/chunkflow_awac_test.py`](../src/openpi/models/chunkflow_awac_test.py)
  checks the equation, extreme temperatures, representable caps, and JIT.

### Eq. (14): actor update

- Actor flow path: `first_action_flow_forward` in
  [`src/openpi/models/pi0.py`](../src/openpi/models/pi0.py) evaluates the executed
  action in the first clean-action position and masks the shape-only future
  tail.
- Objective integration: [`src/openpi/training/chunkflow_awac_train.py`](../src/openpi/training/chunkflow_awac_train.py)
  weights that first-action flow error with the stopped Eq. (13) advantage.
- Tests: [`src/openpi/training/chunkflow_awac_train_test.py`](../src/openpi/training/chunkflow_awac_train_test.py)
  checks random-input sharing, actor/critic gradient separation, and transition
  validation.

Equation 14 is implemented as a **flow-matching surrogate**, not an exact
log-probability or normalized action likelihood.

### Eq. (15): continuity-constrained actor regularization

- Implementation: first/second-order penalties and stopped overlap alignment in
  [`src/openpi/models/chunkflow_losses.py`](../src/openpi/models/chunkflow_losses.py),
  composed through
  [`src/openpi/models/chunkflow_objectives.py`](../src/openpi/models/chunkflow_objectives.py).
- Tests: [`src/openpi/models/chunkflow_losses_test.py`](../src/openpi/models/chunkflow_losses_test.py)
  and [`src/openpi/models/pi0_test.py`](../src/openpi/models/pi0_test.py).
- Scope: the first- and second-order terms directly represent Eq. (15). Boundary
  consistency is a **stopped target-relative residual extension**. It reduces to
  the raw seam term when paired targets use the same coordinate frame and avoids
  penalizing coordinate-origin shifts otherwise.

### Eq. (16): total fine-tuning objective and policy prior

- Composition: [`src/openpi/training/chunkflow_awac_train.py`](../src/openpi/training/chunkflow_awac_train.py)
  reports the supervised structural actor loss, advantage-weighted first-action
  actor surrogate, reference consistency, and the separate Q/V objective.
- Prior surrogate: `reference_consistency_loss` in
  [`src/openpi/models/chunkflow_awac.py`](../src/openpi/models/chunkflow_awac.py)
  compares current and frozen supervised-actor flow velocities under identical
  observation, history, noise, time, and dummy-tail inputs.
- Tests: [`src/openpi/models/chunkflow_awac_test.py`](../src/openpi/models/chunkflow_awac_test.py)
  verifies stopped reference gradients; [`src/openpi/training/chunkflow_awac_train_test.py`](../src/openpi/training/chunkflow_awac_train_test.py)
  verifies shared random inputs and skips the reference forward pass when its
  coefficient is zero.

Equation 16 uses a **reference-consistency surrogate**, not an exact KL. Exact
flow entropy is unavailable and nonzero entropy coefficients are rejected
instead of being reported as an implemented entropy bonus.

## Interpretation boundary

The implementation preserves the paper's algorithmic intent while making the
implicit flow-policy approximations explicit:

- Equation 14: flow-matching surrogate, not an exact log-probability.
- Equation 16: reference-consistency surrogate, not an exact KL or entropy term.
- Boundary consistency: stopped target-relative residual extension for
  coordinate-safe paired training.
- Target-V EMA: stability extension to Equation 12.

These distinctions are part of the public contract and should be preserved in
downstream papers, metrics, and experiment reports.
