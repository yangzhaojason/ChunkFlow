"""Pure numerical losses for supervised ChunkFlow training."""

import numbers

import jax
import jax.numpy as jnp


def flow_endpoint(x_t: jax.Array, velocity: jax.Array, time: jax.Array) -> jax.Array:
    """Extrapolate the flow field at ``(x_t, t)`` to its clean endpoint."""
    if x_t.ndim < 2 or x_t.shape != velocity.shape or time.shape != x_t.shape[:-2]:
        raise ValueError(
            f"incompatible flow shapes: x_t={x_t.shape}, "
            f"velocity={velocity.shape}, time={time.shape}"
        )
    return x_t - time[..., None, None] * velocity


def continuity_penalties(actions: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return per-example first-order L1 and second-order squared penalties."""
    if actions.ndim != 3 or actions.shape[1] <= 0 or actions.shape[2] <= 0:
        raise ValueError(f"actions must have shape [B, T, A], got {actions.shape}")

    if actions.shape[1] < 2:
        first = jnp.zeros(actions.shape[0], dtype=actions.dtype)
    else:
        delta1 = actions[:, 1:] - actions[:, :-1]
        # JAX chooses a nonzero subgradient for abs(0). Select an explicit
        # constant branch at exact matches so a perfectly smooth sequence is
        # a stationary point of this regularizer.
        absolute_delta1 = jnp.where(delta1 == 0, jnp.zeros_like(delta1), jnp.abs(delta1))
        first = jnp.mean(absolute_delta1, axis=(-1, -2))

    if actions.shape[1] < 3:
        second = jnp.zeros(actions.shape[0], dtype=actions.dtype)
    else:
        delta2 = actions[:, 2:] - 2 * actions[:, 1:-1] + actions[:, :-2]
        second = jnp.mean(jnp.square(delta2), axis=(-1, -2))
    return first, second


def boundary_consistency_loss(
    current: jax.Array,
    previous: jax.Array,
    *,
    overlap: int,
    current_target: jax.Array | None = None,
    previous_target: jax.Array | None = None,
) -> jax.Array:
    """Compare aligned adjacent chunks and stop the previous gradient.

    ``overlap`` controls slice shapes and must remain static under ``jax.jit``.
    """
    if current.shape != previous.shape or current.ndim != 3:
        raise ValueError(
            f"paired chunks must share shape [B, L, A], got {current.shape} and {previous.shape}"
        )
    if current.shape[2] <= 0:
        raise ValueError("paired chunks must have a positive action dimension")
    if current.shape[0] <= 0:
        raise ValueError("paired chunks must have a non-empty batch")
    horizon = current.shape[1]
    if (
        isinstance(overlap, bool)
        or not isinstance(overlap, numbers.Integral)
        or not 0 < overlap < horizon
    ):
        raise ValueError(f"overlap must satisfy 0 < O < L, got O={overlap}, L={horizon}")
    overlap = int(overlap)

    if (current_target is None) != (previous_target is None):
        raise ValueError("current_target and previous_target must be provided together")
    if current_target is not None:
        if current_target.shape != current.shape or previous_target.shape != previous.shape:
            raise ValueError("paired target shapes must match endpoint shapes")
        current = current - current_target
        previous = previous - previous_target

    stride = horizon - overlap
    previous_tail = jax.lax.stop_gradient(previous[:, stride : stride + overlap])
    current_head = current[:, :overlap]
    return jnp.mean(jnp.square(current_head - previous_tail))
