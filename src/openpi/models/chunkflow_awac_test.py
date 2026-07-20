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
