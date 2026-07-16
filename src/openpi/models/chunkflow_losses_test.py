import jax
import jax.numpy as jnp
import pytest

from openpi.models.chunkflow_losses import boundary_consistency_loss
from openpi.models.chunkflow_losses import continuity_penalties
from openpi.models.chunkflow_losses import flow_endpoint


def test_flow_endpoint_extrapolates_to_clean_action():
    action = jnp.array([[[1.0, -2.0]]])
    noise = jnp.array([[[5.0, 6.0]]])
    time = jnp.array([0.25])
    velocity = noise - action
    x_t = time[:, None, None] * noise + (1.0 - time[:, None, None]) * action

    assert jnp.allclose(flow_endpoint(x_t, velocity, time), action)


def test_flow_endpoint_rejects_incompatible_shapes():
    with pytest.raises(ValueError, match="flow shapes"):
        flow_endpoint(
            jnp.zeros((2, 3, 4)),
            jnp.zeros((2, 3, 5)),
            jnp.zeros((2,)),
        )
    with pytest.raises(ValueError, match="flow shapes"):
        flow_endpoint(
            jnp.zeros((2, 3, 4)),
            jnp.zeros((2, 3, 4)),
            jnp.zeros((2, 1)),
        )


def test_continuity_matches_known_values():
    actions = jnp.array([[[0.0, 0.0], [1.0, 2.0], [3.0, 6.0]]])

    first, second = continuity_penalties(actions)

    assert first[0] == pytest.approx(2.25)
    assert second[0] == pytest.approx(2.5)


def test_continuity_is_zero_when_derivative_is_undefined():
    first, second = continuity_penalties(jnp.ones((2, 1, 3)))

    assert jnp.array_equal(first, jnp.zeros(2))
    assert jnp.array_equal(second, jnp.zeros(2))


def test_first_order_continuity_uses_zero_subgradient_at_exact_match():
    actions = jnp.ones((1, 3, 1))

    gradient = jax.grad(lambda value: jnp.sum(continuity_penalties(value)[0]))(actions)

    assert jnp.array_equal(gradient, jnp.zeros_like(actions))


def test_continuity_rejects_non_batched_action_chunks():
    with pytest.raises(ValueError, match=r"\[B, T, A\]"):
        continuity_penalties(jnp.ones((3, 2)))


def test_boundary_uses_aligned_tail_and_stops_previous_gradient():
    previous = jnp.array([[[0.0], [1.0], [2.0], [3.0]]])
    current = jnp.array([[[4.0], [5.0], [6.0], [7.0]]])

    loss = boundary_consistency_loss(current, previous, overlap=2)

    assert loss == pytest.approx(4.0)
    current_grad, previous_grad = jax.grad(
        lambda cur, prev: boundary_consistency_loss(cur, prev, overlap=2),
        argnums=(0, 1),
    )(current, previous)
    assert jnp.any(current_grad != 0)
    assert jnp.all(previous_grad == 0)


def test_boundary_residuals_cancel_state_relative_coordinate_origins():
    previous_target = jnp.array([[[0.0], [1.0], [2.0], [3.0]]])
    current_target = jnp.array([[[12.0], [13.0], [14.0], [15.0]]])
    previous_endpoint = previous_target + 0.5
    current_endpoint = current_target + 0.5

    loss = boundary_consistency_loss(
        current_endpoint,
        previous_endpoint,
        overlap=2,
        current_target=current_target,
        previous_target=previous_target,
    )

    assert loss == 0


@pytest.mark.parametrize("overlap", [0, 4, -1, True, 1.5])
def test_boundary_rejects_invalid_overlap(overlap):
    chunk = jnp.zeros((1, 4, 2))

    with pytest.raises(ValueError, match="overlap"):
        boundary_consistency_loss(chunk, chunk, overlap=overlap)


def test_boundary_requires_paired_targets_with_matching_shapes():
    chunk = jnp.zeros((1, 4, 2))

    with pytest.raises(ValueError, match="provided together"):
        boundary_consistency_loss(chunk, chunk, overlap=2, current_target=chunk)
    with pytest.raises(ValueError, match="target shapes"):
        boundary_consistency_loss(
            chunk,
            chunk,
            overlap=2,
            current_target=jnp.zeros((1, 3, 2)),
            previous_target=jnp.zeros((1, 3, 2)),
        )


def test_boundary_rejects_empty_action_dimension():
    chunk = jnp.zeros((1, 4, 0))

    with pytest.raises(ValueError, match="action dimension"):
        boundary_consistency_loss(chunk, chunk, overlap=2)


def test_boundary_rejects_empty_batch():
    chunk = jnp.zeros((0, 4, 2))

    with pytest.raises(ValueError, match="batch"):
        boundary_consistency_loss(chunk, chunk, overlap=2)


def test_boundary_loss_is_jittable_with_static_overlap():
    chunk = jnp.zeros((1, 4, 2))
    compiled = jax.jit(boundary_consistency_loss, static_argnames=("overlap",))

    assert compiled(chunk, chunk, overlap=2) == 0
