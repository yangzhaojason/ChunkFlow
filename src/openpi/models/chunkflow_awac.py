"""Pure numerical objectives for step-wise ChunkFlow AWAC.

Scalar configuration keyword arguments are Python-validated and must remain
static when these helpers are transformed with :func:`jax.jit`.
"""

import math
import numbers

import jax
import jax.numpy as jnp
import numpy as np


def _finite_real(value: float, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _representable_cap(value: float, *, dtype) -> float:
    dtype = np.dtype(dtype)
    target = min(value, float(jnp.finfo(dtype).max))
    rounded = np.asarray(target, dtype=dtype)
    if float(rounded) > target:
        rounded = np.nextafter(rounded, np.asarray(-np.inf, dtype=dtype))
    return float(rounded)


def td_targets(
    reward: jax.Array,
    continuation: jax.Array,
    next_value: jax.Array,
    *,
    gamma: float,
) -> jax.Array:
    """Return one-step temporal-difference targets.

    ``gamma`` must be static under :func:`jax.jit`.
    """
    if reward.ndim != 1 or continuation.shape != reward.shape or next_value.shape != reward.shape:
        raise ValueError("reward, continuation, and next_value must share shape [B]")
    gamma = _finite_real(gamma, name="gamma")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be within [0, 1]")
    return reward + gamma * continuation * jax.lax.stop_gradient(next_value)


def expectile_value_loss(
    q_value: jax.Array,
    value: jax.Array,
    *,
    expectile: float,
) -> tuple[jax.Array, jax.Array]:
    """Return the mean expectile loss and stopped-Q residual.

    ``expectile`` must be static under :func:`jax.jit`.
    """
    if q_value.ndim != 1 or value.shape != q_value.shape:
        raise ValueError("q_value and value must share shape [B]")
    if q_value.shape[0] == 0:
        raise ValueError("q_value and value must be non-empty")
    expectile = _finite_real(expectile, name="expectile")
    if not 0.0 < expectile < 1.0:
        raise ValueError("expectile must be within (0, 1)")

    residual = jax.lax.stop_gradient(q_value) - value
    coefficient = jnp.where(residual < 0, 1.0 - expectile, expectile)
    return jnp.mean(coefficient * jnp.square(residual)), residual


def clipped_advantage_weights(
    q_value: jax.Array,
    value: jax.Array,
    *,
    temperature: float,
    wmax: float,
) -> tuple[jax.Array, jax.Array]:
    """Return nonnegative exponentially weighted stopped advantages.

    ``temperature`` and ``wmax`` must be static under :func:`jax.jit`.
    """
    if q_value.ndim != 1 or value.shape != q_value.shape:
        raise ValueError("q_value and value must share shape [B]")
    temperature = _finite_real(temperature, name="temperature")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    wmax = _finite_real(wmax, name="wmax")
    if wmax < 1.0:
        raise ValueError("wmax must be at least 1")

    advantage_dtype = jnp.result_type(q_value.dtype, value.dtype)
    if not jnp.issubdtype(advantage_dtype, jnp.floating):
        raise ValueError("q_value and value must produce a floating advantage dtype")
    advantage = jax.lax.stop_gradient(q_value - value)
    effective_cap = jnp.asarray(
        _representable_cap(wmax, dtype=advantage.dtype),
        dtype=advantage.dtype,
    )
    nonpositive = advantage <= 0
    safe_advantage = jnp.where(nonpositive, jnp.ones_like(advantage), advantage)
    positive_exponent = jnp.minimum(safe_advantage / temperature, jnp.log(effective_cap))
    capped_exponent = jnp.where(nonpositive, jnp.zeros_like(advantage), positive_exponent)
    weights = jnp.minimum(jnp.exp(capped_exponent), effective_cap)
    return weights, advantage


def reference_consistency_loss(
    current_velocity: jax.Array,
    reference_velocity: jax.Array,
) -> jax.Array:
    """Return mean-square consistency loss against a stopped reference."""
    if current_velocity.shape != reference_velocity.shape:
        raise ValueError("current_velocity and reference_velocity must share shape")
    if current_velocity.size == 0:
        raise ValueError("current_velocity and reference_velocity must be non-empty")
    difference = current_velocity - jax.lax.stop_gradient(reference_velocity)
    return jnp.mean(jnp.square(difference))
