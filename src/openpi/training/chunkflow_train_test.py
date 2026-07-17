import jax.numpy as jnp
import pytest

from openpi.training.chunkflow_batch import PairedChunkBatch
from openpi.training.chunkflow_train import batch_observation
from openpi.training.chunkflow_train import batch_with_step
from openpi.training.chunkflow_train import combine_supervised_losses
from openpi.training.chunkflow_train import coordinate_aligned_predicted_history
from openpi.training.chunkflow_train import share_boundary_noise


def test_supervised_loss_combines_per_step_flow_continuity_and_boundary():
    total, metrics = combine_supervised_losses(
        jnp.array([[1.0, 3.0]]),
        first_order=jnp.array([2.0]),
        second_order=jnp.array([4.0]),
        boundary=jnp.array(5.0),
        first_weight=0.1,
        second_weight=0.2,
        boundary_weight=0.3,
    )

    assert total == pytest.approx(4.5)
    assert metrics["loss/flow"] == pytest.approx(2.0)
    assert metrics["loss/continuity_first"] == pytest.approx(2.0)
    assert metrics["loss/continuity_second"] == pytest.approx(4.0)
    assert metrics["loss/boundary"] == pytest.approx(5.0)


def test_zero_auxiliary_weights_preserve_mean_flow_loss():
    total, _ = combine_supervised_losses(
        jnp.array([[1.0, 3.0], [5.0, 7.0]]),
        first_order=jnp.array([100.0, 100.0]),
        second_order=jnp.array([100.0, 100.0]),
        boundary=jnp.array(100.0),
        first_weight=0.0,
        second_weight=0.0,
        boundary_weight=0.0,
    )

    assert total == pytest.approx(4.0)


def test_predicted_history_transfers_residual_into_current_coordinate_frame():
    previous_actions = jnp.array([[[0.0], [1.0], [2.0], [3.0]]])
    previous_endpoint = previous_actions + 0.5
    current_history = jnp.array([[[10.0], [11.0]]])

    predicted = coordinate_aligned_predicted_history(
        previous_endpoint,
        previous_actions,
        current_history,
        stride=3,
    )

    assert predicted.tolist() == [[[10.5], [11.5]]]


def test_predicted_history_keeps_the_current_history_dtype():
    previous_actions = jnp.zeros((1, 4, 1), dtype=jnp.bfloat16)
    previous_endpoint = previous_actions.astype(jnp.float32) + 0.5
    current_history = jnp.ones((1, 2, 1), dtype=jnp.bfloat16)

    predicted = coordinate_aligned_predicted_history(
        previous_endpoint,
        previous_actions,
        current_history,
        stride=3,
    )

    assert predicted.dtype == current_history.dtype


def test_boundary_noise_is_shared_only_at_the_aligned_seam():
    previous = jnp.zeros((1, 4, 1))
    current = jnp.arange(4, dtype=jnp.float32).reshape(1, 4, 1)

    aligned = share_boundary_noise(previous, current, overlap=2)

    assert aligned[:, :2].tolist() == [[[0.0], [0.0]]]
    assert aligned[:, 2:].tolist() == current[:, :2].tolist()


def test_zero_overlap_keeps_previous_noise_unchanged():
    previous = jnp.arange(4, dtype=jnp.float32).reshape(1, 4, 1)
    current = jnp.zeros_like(previous)

    assert jnp.array_equal(share_boundary_noise(previous, current, overlap=0), previous)


def test_paired_batch_uses_current_observation_and_optimizer_step():
    batch = PairedChunkBatch(
        previous_observation={"state": jnp.array([0.0])},
        previous_actions=jnp.zeros((1, 2, 1)),
        observation={"state": jnp.array([1.0])},
        actions=jnp.ones((1, 2, 1)),
        step=jnp.array([0]),
    )

    prepared = batch_with_step(batch, jnp.array(7, dtype=jnp.int32))

    assert batch_observation(prepared)["state"].item() == 1.0
    assert prepared.step.shape == ()
    assert prepared.step.item() == 7


def test_legacy_batch_observation_is_unchanged():
    observation = {"state": jnp.array([2.0])}

    assert batch_observation((observation, jnp.zeros((1, 2, 1)))) is observation
