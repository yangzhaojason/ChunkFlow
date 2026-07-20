"""Pure numerical objectives for step-wise ChunkFlow AWAC."""

import math
import numbers

import jax
import jax.numpy as jnp


def _finite_real(value: float, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be finite")
    return float(value)


def td_targets(
    reward: jax.Array,
    continuation: jax.Array,
    next_value: jax.Array,
    *,
    gamma: float,
) -> jax.Array:
    """Return one-step temporal-difference targets."""
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
    """Return the mean expectile loss and stopped-Q residual."""
    if q_value.ndim != 1 or value.shape != q_value.shape:
        raise ValueError("q_value and value must share shape [B]")
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
    """Return nonnegative exponentially weighted stopped advantages."""
    if q_value.ndim != 1 or value.shape != q_value.shape:
        raise ValueError("q_value and value must share shape [B]")
    temperature = _finite_real(temperature, name="temperature")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    wmax = _finite_real(wmax, name="wmax")
    if wmax < 1.0:
        raise ValueError("wmax must be at least 1")

    advantage = jax.lax.stop_gradient(q_value - value)
    capped_exponent = jnp.minimum(
        jnp.maximum(advantage, 0.0) / temperature,
        jnp.log(wmax),
    )
    return jnp.exp(capped_exponent), advantage


def reference_consistency_loss(
    current_velocity: jax.Array,
    reference_velocity: jax.Array,
) -> jax.Array:
    """Return mean-square consistency loss against a stopped reference."""
    if current_velocity.shape != reference_velocity.shape:
        raise ValueError("current_velocity and reference_velocity must share shape")
    difference = current_velocity - jax.lax.stop_gradient(reference_velocity)
    return jnp.mean(jnp.square(difference))
