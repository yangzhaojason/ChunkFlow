import jax
import jax.numpy as jnp
import pytest

from openpi.models.chunkflow_awac import clipped_advantage_weights
from openpi.models.chunkflow_awac import expectile_value_loss
from openpi.models.chunkflow_awac import reference_consistency_loss
from openpi.models.chunkflow_awac import td_targets


def test_td_target_applies_gamma_once_and_stops_at_terminal():
    result = td_targets(
        jnp.array([1.0, 2.0]),
        jnp.array([1.0, 0.0]),
        jnp.array([10.0, 100.0]),
        gamma=0.9,
    )
    assert result.tolist() == pytest.approx([10.0, 2.0])


def test_expectile_matches_positive_negative_and_zero_residuals():
    loss, residual = expectile_value_loss(
        jnp.array([3.0, 1.0, 2.0]),
        jnp.array([1.0, 3.0, 2.0]),
        expectile=0.7,
    )
    assert residual.tolist() == pytest.approx([2.0, -2.0, 0.0])
    assert loss == pytest.approx((0.7 * 4 + 0.3 * 4) / 3)


def test_advantage_weights_match_equation_and_are_finite_at_extremes():
    weights, advantage = clipped_advantage_weights(
        jnp.array([-2.0, 0.0, 0.1, 1e30]),
        jnp.zeros(4),
        temperature=0.1,
        wmax=20.0,
    )
    assert advantage.tolist() == pytest.approx([-2.0, 0.0, 0.1, 1e30])
    assert weights[:3].tolist() == pytest.approx([1.0, 1.0, jnp.e])
    assert weights[3] == 20.0
    assert jnp.all(jnp.isfinite(weights))


def test_reference_consistency_stops_reference_gradient():
    current = jnp.array([[1.0, 3.0]])
    reference = jnp.array([[0.0, 1.0]])
    assert reference_consistency_loss(reference, reference) == pytest.approx(0.0)
    assert reference_consistency_loss(current, reference) == pytest.approx(2.5)
    current_grad, reference_grad = jax.grad(
        lambda cur, ref: reference_consistency_loss(cur, ref),  # noqa: PLW0108
        argnums=(0, 1),
    )(current, reference)
    assert jnp.any(current_grad != 0)
    assert jnp.all(reference_grad == 0)


@pytest.mark.parametrize(
    ("dtype", "temperature"),
    [
        (jnp.float16, 1e-8),
        (jnp.float32, 1e-50),
    ],
)
def test_advantage_weights_stay_finite_when_temperature_underflows(dtype, temperature):
    q_value = jnp.asarray([-1.0, 0.0, 1.0], dtype=dtype)

    weights, _ = clipped_advantage_weights(
        q_value,
        jnp.zeros_like(q_value),
        temperature=temperature,
        wmax=20.0,
    )

    cap = jnp.asarray(20.0, dtype=dtype)
    assert jnp.all(jnp.isfinite(weights))
    assert jnp.array_equal(weights[:2], jnp.ones(2, dtype=dtype))
    assert jnp.isfinite(cap)
    assert weights[2] <= cap


def test_bfloat16_advantage_weights_do_not_round_above_wmax():
    q_value = jnp.asarray([-1.0, 0.0, 100.0], dtype=jnp.bfloat16)

    weights, _ = clipped_advantage_weights(
        q_value,
        jnp.zeros_like(q_value),
        temperature=0.1,
        wmax=20.0,
    )

    representable_wmax = jnp.asarray(20.0, dtype=jnp.bfloat16)
    assert jnp.all(jnp.isfinite(weights))
    assert jnp.all(weights <= representable_wmax)


@pytest.mark.parametrize(
    "wmax",
    [float(jnp.finfo(jnp.float16).max), 1e100],
    ids=("dtype-max", "larger-than-dtype-max"),
)
def test_float16_advantage_weights_cap_unrepresentable_wmax(wmax):
    q_value = jnp.asarray([1000.0], dtype=jnp.float16)

    weights, _ = clipped_advantage_weights(
        q_value,
        jnp.zeros_like(q_value),
        temperature=0.1,
        wmax=wmax,
    )

    dtype_cap = jnp.asarray(jnp.finfo(jnp.float16).max, dtype=jnp.float16)
    assert weights.dtype == jnp.float16
    assert jnp.all(jnp.isfinite(weights))
    assert jnp.all(weights <= dtype_cap)


def test_expectile_rejects_empty_batch():
    empty = jnp.zeros((0,), dtype=jnp.float32)

    with pytest.raises(ValueError, match="non-empty"):
        expectile_value_loss(empty, empty, expectile=0.7)


def test_reference_consistency_rejects_empty_inputs():
    empty = jnp.zeros((0,), dtype=jnp.float32)

    with pytest.raises(ValueError, match="non-empty"):
        reference_consistency_loss(empty, empty)


def test_advantage_weights_reject_non_floating_advantage_dtype():
    values = jnp.array([1, 2], dtype=jnp.int32)

    with pytest.raises(ValueError, match="floating"):
        clipped_advantage_weights(values, values, temperature=0.1, wmax=20.0)


def test_td_targets_is_jittable_with_static_gamma():
    compiled = jax.jit(td_targets, static_argnames=("gamma",))

    result = compiled(
        jnp.array([1.0]),
        jnp.array([1.0]),
        jnp.array([2.0]),
        gamma=0.5,
    )

    assert result.tolist() == pytest.approx([2.0])


def test_expectile_loss_is_jittable_with_static_expectile():
    compiled = jax.jit(expectile_value_loss, static_argnames=("expectile",))

    loss, residual = compiled(
        jnp.array([2.0]),
        jnp.array([1.0]),
        expectile=0.7,
    )

    assert loss == pytest.approx(0.7)
    assert residual.tolist() == pytest.approx([1.0])


def test_advantage_weights_are_jittable_with_static_configuration():
    compiled = jax.jit(
        clipped_advantage_weights,
        static_argnames=("temperature", "wmax"),
    )

    weights, advantage = compiled(
        jnp.array([0.1]),
        jnp.array([0.0]),
        temperature=0.1,
        wmax=20.0,
    )

    assert weights.tolist() == pytest.approx([jnp.e])
    assert advantage.tolist() == pytest.approx([0.1])
