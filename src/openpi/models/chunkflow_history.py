"""Pure helpers for robust executed-action history conditioning."""

import math
import numbers

import jax
import jax.numpy as jnp


def _require_nonnegative_integer(value, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
        raise ValueError(f"invalid scheduled-sampling configuration: {name} must be non-negative")
    return int(value)


def _require_probability(value, *, name: str, context: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"invalid {context}: {name} must be finite and within [0, 1]")
    return float(value)


def validate_history(history: jax.Array, mask: jax.Array, *, action_dim: int) -> None:
    """Validate a batched history tensor and its episode-validity mask."""
    if isinstance(action_dim, bool) or not isinstance(action_dim, numbers.Integral) or action_dim <= 0:
        raise ValueError("history action_dim must be a positive integer")
    if history.ndim != 3 or history.shape[-1] != action_dim or history.shape[0] <= 0:
        raise ValueError(
            f"history must have shape [B, P, {action_dim}] with B > 0, got {history.shape}"
        )
    if not jnp.issubdtype(history.dtype, jnp.floating):
        raise ValueError(f"history must have a real floating dtype, got {history.dtype}")
    if mask.shape != history.shape[:-1] or mask.dtype != jnp.bool_:
        raise ValueError(
            f"history mask must be bool [B, P], got shape={mask.shape}, dtype={mask.dtype}"
        )


def scheduled_sampling_alpha(
    step: int | jax.Array,
    *,
    warmup_steps: int,
    ramp_steps: int,
    max_alpha: float,
) -> jax.Array:
    """Linearly ramp scheduled-sampling weight after an initial warmup.

    The three configuration arguments must be static under ``jax.jit``; only
    ``step`` is intended to be traced.
    """
    warmup_steps = _require_nonnegative_integer(warmup_steps, name="warmup_steps")
    ramp_steps = _require_nonnegative_integer(ramp_steps, name="ramp_steps")
    if ramp_steps == 0:
        raise ValueError("invalid scheduled-sampling configuration: ramp_steps must be positive")
    max_alpha = _require_probability(
        max_alpha,
        name="max_alpha",
        context="scheduled-sampling configuration",
    )
    if isinstance(step, bool):
        raise ValueError("scheduled-sampling step must be an integer scalar")
    if isinstance(step, numbers.Integral):
        step_value = int(step)
        if -(2**31) <= step_value <= 2**31 - 1:
            step_array = jnp.asarray(step_value, dtype=jnp.int32)
        elif 0 <= step_value <= 2**32 - 1:
            step_array = jnp.asarray(step_value, dtype=jnp.uint32)
        else:
            raise ValueError("scheduled-sampling step must fit int32 or uint32")
    else:
        step_array = jnp.asarray(step)
    if step_array.shape != () or not jnp.issubdtype(step_array.dtype, jnp.integer):
        raise ValueError("scheduled-sampling step must be an integer scalar")
    dtype_limit = int(jnp.iinfo(step_array.dtype).max)
    if warmup_steps > dtype_limit:
        raise ValueError(
            "invalid scheduled-sampling configuration: warmup_steps exceeds the step dtype"
        )
    # Clamp and subtract in the original integer domain. This avoids unsigned
    # underflow and preserves one-step differences beyond float32's exact range.
    warmup_array = jnp.asarray(warmup_steps, dtype=step_array.dtype)
    elapsed_steps = jnp.maximum(step_array, warmup_array) - warmup_array
    progress = jnp.clip(elapsed_steps.astype(jnp.float32) / float(ramp_steps), 0.0, 1.0)
    return progress * max_alpha


def corrupt_history(
    rng: jax.Array,
    clean: jax.Array,
    valid_mask: jax.Array,
    predicted: jax.Array,
    *,
    prediction_mask: jax.Array | None = None,
    noise_std: float,
    dropout_probability: float,
    alpha: float | jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Corrupt demo history and mix in stopped EMA-actor predictions.

    ``noise_std`` and ``dropout_probability`` are static configuration values
    under ``jax.jit``. ``alpha`` may be a dynamic scalar from
    :func:`scheduled_sampling_alpha`. For one eager/JIT contract, all alpha
    inputs are projected to a finite convex weight: NaN and negative infinity
    fall back to zero, positive infinity to one, then finite values are clipped.
    """
    if getattr(clean, "ndim", None) != 3:
        raise ValueError(f"history must have shape [B, P, A], got {getattr(clean, 'shape', None)}")
    validate_history(clean, valid_mask, action_dim=clean.shape[-1])
    if predicted.shape != clean.shape:
        raise ValueError(f"predicted history shape {predicted.shape} does not match {clean.shape}")
    if predicted.dtype != clean.dtype:
        raise ValueError(
            f"predicted history dtype {predicted.dtype} does not match clean dtype {clean.dtype}"
        )
    if prediction_mask is None:
        prediction_mask = valid_mask
    if prediction_mask.shape != valid_mask.shape or prediction_mask.dtype != jnp.bool_:
        raise ValueError("prediction_mask must be bool [B, P]")
    prediction_mask = jnp.logical_and(prediction_mask, valid_mask)
    if (
        isinstance(noise_std, bool)
        or not isinstance(noise_std, numbers.Real)
        or not math.isfinite(float(noise_std))
        or noise_std < 0
    ):
        raise ValueError("invalid history corruption configuration: noise_std must be finite and non-negative")
    dropout_probability = _require_probability(
        dropout_probability,
        name="dropout_probability",
        context="history corruption configuration",
    )
    alpha_array = jnp.asarray(alpha, dtype=clean.dtype)
    if alpha_array.shape != ():
        raise ValueError("invalid history corruption configuration: alpha must be a scalar")
    alpha_array = jnp.nan_to_num(alpha_array, nan=0.0, posinf=1.0, neginf=0.0)
    alpha_array = jnp.clip(alpha_array, 0.0, 1.0)

    noise_rng, dropout_rng = jax.random.split(rng)
    noise = jax.random.normal(noise_rng, clean.shape, dtype=clean.dtype)
    noisy_demo = clean + float(noise_std) * noise
    dropped = jax.random.bernoulli(
        dropout_rng,
        p=dropout_probability,
        shape=valid_mask.shape,
    )
    corrupted_demo = jnp.where(dropped[..., None], jnp.zeros_like(noisy_demo), noisy_demo)
    stopped_prediction = jax.lax.stop_gradient(predicted)
    effective_alpha = jnp.where(prediction_mask, alpha_array, jnp.zeros_like(alpha_array))
    effective_alpha = effective_alpha[..., None]
    mixed_candidate = (
        (1.0 - effective_alpha) * corrupted_demo + effective_alpha * stopped_prediction
    )
    mixed = jnp.where(effective_alpha <= 0.0, corrupted_demo, mixed_candidate)
    mixed = jnp.where(effective_alpha >= 1.0, stopped_prediction, mixed)
    mixed = jnp.where(valid_mask[..., None], mixed, jnp.zeros_like(mixed))
    return mixed, valid_mask
