"""Pure objective helpers shared by ChunkFlow models and training loops."""

import numbers

import jax
import jax.numpy as jnp


def share_boundary_noise(
    previous_noise: jax.Array,
    current_noise: jax.Array,
    *,
    overlap: int,
) -> jax.Array:
    """Copy current-head flow noise into the aligned previous tail."""

    if previous_noise.shape != current_noise.shape or previous_noise.ndim != 3:
        raise ValueError("paired noise must share shape [B, L, A]")
    horizon = previous_noise.shape[1]
    if (
        isinstance(overlap, bool)
        or not isinstance(overlap, numbers.Integral)
        or not 0 <= overlap < horizon
    ):
        raise ValueError("overlap must satisfy 0 <= overlap < horizon")
    if overlap == 0:
        return previous_noise
    stride = horizon - overlap
    return previous_noise.at[:, stride:].set(current_noise[:, :overlap])


def coordinate_aligned_predicted_history(
    previous_endpoint: jax.Array,
    previous_actions: jax.Array,
    current_history: jax.Array,
    *,
    stride: int,
) -> tuple[jax.Array, jax.Array]:
    """Transfer the previous prediction residual into current action coordinates."""

    if previous_endpoint.shape != previous_actions.shape or previous_endpoint.ndim != 3:
        raise ValueError("previous endpoint and actions must share shape [B, L, A]")
    if current_history.ndim != 3:
        raise ValueError("current_history must have shape [B, P, A]")
    if (
        current_history.shape[0] != previous_actions.shape[0]
        or current_history.shape[2] != previous_actions.shape[2]
    ):
        raise ValueError("current_history batch and action dimensions must match previous actions")
    horizon = previous_actions.shape[1]
    if not 0 <= stride <= horizon:
        raise ValueError("stride must satisfy 0 <= stride <= horizon")

    history_length = current_history.shape[1]
    covered = min(history_length, stride)
    prediction_mask = jnp.zeros(current_history.shape[:2], dtype=jnp.bool_)
    if covered == 0:
        return current_history, prediction_mask

    previous_slice = slice(stride - covered, stride)
    history_slice = slice(history_length - covered, history_length)
    residual = (
        previous_endpoint[:, previous_slice] - previous_actions[:, previous_slice]
    ).astype(current_history.dtype)
    predicted = current_history.at[:, history_slice].add(residual)
    prediction_mask = prediction_mask.at[:, history_slice].set(True)
    return predicted, prediction_mask


def combine_supervised_losses(
    per_step_flow: jax.Array,
    *,
    first_order: jax.Array,
    second_order: jax.Array,
    boundary: jax.Array,
    first_weight: float,
    second_weight: float,
    boundary_weight: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Reduce and combine flow, continuity, and paired-boundary losses."""

    flow = jnp.mean(per_step_flow)
    first = jnp.mean(first_order)
    second = jnp.mean(second_order)
    boundary = jnp.mean(boundary)
    total = flow + first_weight * first + second_weight * second + boundary_weight * boundary
    return total, {
        "loss/flow": flow,
        "loss/continuity_first": first,
        "loss/continuity_second": second,
        "loss/boundary": boundary,
    }
