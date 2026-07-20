import dataclasses

import numpy as np
import pytest

from openpi.training.chunkflow_rlds import build_tf_paired_chunks
from openpi.training.chunkflow_rlds import build_tf_step_transitions
from openpi.training.chunkflow_rlds import episode_step_transition_arrays


def test_episode_step_transition_arrays_include_terminal_and_shift_history():
    result = episode_step_transition_arrays(
        np.array([[10.0], [11.0], [12.0]], np.float32),
        np.array([0.0, 0.0, 1.0], np.float32),
        np.array([1.0, 1.0, 0.0], np.float32),
        history_length=2,
    )

    np.testing.assert_array_equal(result.indices, [0, 1, 2])
    np.testing.assert_array_equal(result.next_indices, [1, 2, 2])
    np.testing.assert_array_equal(result.actions[:, 0], [10.0, 11.0, 12.0])
    np.testing.assert_array_equal(result.reward, [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(result.continuation, [1.0, 1.0, 0.0])
    np.testing.assert_array_equal(result.history[:, :, 0], [[0.0, 0.0], [0.0, 10.0], [10.0, 11.0]])
    np.testing.assert_array_equal(result.next_history[:, :, 0], [[0.0, 10.0], [10.0, 11.0], [11.0, 12.0]])
    np.testing.assert_array_equal(result.history_mask, [[False, False], [False, True], [True, True]])
    np.testing.assert_array_equal(result.next_history_mask, [[False, True], [True, True], [True, True]])


def test_episode_step_transition_arrays_requires_terminal_continuation():
    with pytest.raises(ValueError, match="terminal"):
        episode_step_transition_arrays(
            np.ones((2, 1), dtype=np.float32),
            np.zeros((2,), dtype=np.float32),
            np.ones((2,), dtype=np.float32),
            history_length=0,
        )


def test_episode_step_transition_arrays_handles_single_step_and_zero_history():
    result = episode_step_transition_arrays(
        np.array([[3.0, 4.0]], dtype=np.float64),
        np.array([2.0], dtype=np.float32),
        np.array([0.0], dtype=np.float16),
        history_length=0,
    )

    np.testing.assert_array_equal(result.indices, [0])
    np.testing.assert_array_equal(result.next_indices, [0])
    assert result.history.shape == (1, 0, 2)
    assert result.next_history.shape == (1, 0, 2)
    assert result.history_mask.shape == (1, 0)
    assert result.next_history_mask.shape == (1, 0)
    assert result.actions.dtype == np.float64
    assert result.reward.dtype == np.float32
    assert result.continuation.dtype == np.float16


@pytest.mark.parametrize(
    ("actions", "rewards", "continuation", "history_length", "message"),
    [
        (np.empty((0, 1), np.float32), np.empty((0,), np.float32), np.empty((0,), np.float32), 0, "nonempty"),
        (np.ones((2,), np.float32), np.ones((2,), np.float32), np.array([1.0, 0.0], np.float32), 0, "rank 2"),
        (np.ones((2, 1), np.int32), np.ones((2,), np.float32), np.array([1.0, 0.0], np.float32), 0, "floating"),
        (np.ones((2, 1), np.complex64), np.ones((2,), np.float32), np.array([1.0, 0.0], np.float32), 0, "real floating"),
        (np.array([[1.0], [np.inf]], np.float32), np.ones((2,), np.float32), np.array([1.0, 0.0], np.float32), 0, "finite"),
        (np.ones((2, 1), np.float32), np.ones((2, 1), np.float32), np.array([1.0, 0.0], np.float32), 0, "rank 1"),
        (np.ones((2, 1), np.float32), np.ones((1,), np.float32), np.array([1.0, 0.0], np.float32), 0, "length"),
        (np.ones((2, 1), np.float32), np.ones((2,), np.int32), np.array([1.0, 0.0], np.float32), 0, "floating"),
        (np.ones((2, 1), np.float32), np.array([0.0, np.nan], np.float32), np.array([1.0, 0.0], np.float32), 0, "finite"),
        (np.ones((2, 1), np.float32), np.ones((2,), np.float32), np.array([[1.0], [0.0]], np.float32), 0, "rank 1"),
        (np.ones((2, 1), np.float32), np.ones((2,), np.float32), np.array([0.0], np.float32), 0, "length"),
        (np.ones((2, 1), np.float32), np.ones((2,), np.float32), np.array([1, 0], np.int32), 0, "floating"),
        (np.ones((2, 1), np.float32), np.ones((2,), np.float32), np.array([1.1, 0.0], np.float32), 0, r"\[0, 1\]"),
        (np.ones((2, 1), np.float32), np.ones((2,), np.float32), np.array([np.nan, 0.0], np.float32), 0, "finite"),
        (np.ones((2, 1), np.float32), np.ones((2,), np.float32), np.array([1.0, 0.0], np.float32), -1, "history_length"),
        (np.ones((2, 1), np.float32), np.ones((2,), np.float32), np.array([1.0, 0.0], np.float32), 1.5, "integer"),
        (np.ones((2, 1), np.float32), np.ones((2,), np.float32), np.array([1.0, 0.0], np.float32), True, "integer"),
    ],
)
def test_episode_step_transition_arrays_validates_inputs(actions, rewards, continuation, history_length, message):
    with pytest.raises(ValueError, match=message):
        episode_step_transition_arrays(actions, rewards, continuation, history_length)


def test_episode_step_transition_arrays_does_not_mutate_inputs_or_expose_writable_dataclass():
    actions = np.array([[1.0], [2.0]], dtype=np.float32)
    rewards = np.array([0.0, 1.0], dtype=np.float32)
    continuation = np.array([1.0, 0.0], dtype=np.float32)
    originals = tuple(value.copy() for value in (actions, rewards, continuation))

    result = episode_step_transition_arrays(actions, rewards, continuation, 1)

    for value, original in zip((actions, rewards, continuation), originals, strict=True):
        np.testing.assert_array_equal(value, original)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.indices = np.array([], dtype=np.int64)


@pytest.mark.parametrize("history_length", [-1, 1.5, True])
def test_tf_step_transition_static_history_is_validated_without_tensorflow(history_length):
    with pytest.raises(ValueError, match="history_length"):
        build_tf_step_transitions({}, history_length=history_length, tf=object())


@pytest.mark.parametrize(
    ("horizon", "stride", "history_length"),
    [(0, 1, 0), (4, 0, 0), (4, 5, 0), (4, 2, -1), (4, 2, 5), (True, 1, 0)],
)
def test_tf_paired_chunk_static_geometry_is_validated_without_tensorflow(horizon, stride, history_length):
    with pytest.raises(ValueError, match=r"horizon|stride|history_length"):
        build_tf_paired_chunks(
            {},
            horizon=horizon,
            stride=stride,
            history_length=history_length,
            tf=object(),
        )


def test_tf_step_transitions_match_numpy_reference_when_tensorflow_is_available():
    tf = pytest.importorskip("tensorflow")
    actions = np.array([[10.0], [11.0], [12.0]], dtype=np.float32)
    rewards = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    discounts = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    expected = episode_step_transition_arrays(actions, rewards, discounts, history_length=2)
    traj = {
        "actions": tf.constant(actions),
        "rewards": tf.constant(rewards),
        "discounts": tf.constant(discounts),
        "observation": {"state": tf.constant([[1.0], [2.0], [3.0]])},
        "prompt": tf.constant(["a", "b", "c"]),
        "episode_id": tf.constant(7, tf.int32),
    }

    result = build_tf_step_transitions(traj, history_length=2, tf=tf)

    np.testing.assert_array_equal(result["frame_index"].numpy(), expected.indices)
    np.testing.assert_array_equal(result["current"]["action_history"].numpy(), expected.history)
    np.testing.assert_array_equal(result["next"]["action_history"].numpy(), expected.next_history)
    np.testing.assert_array_equal(result["next"]["observation"]["state"].numpy()[:, 0], [2.0, 3.0, 3.0])
    np.testing.assert_array_equal(result["episode_id"].numpy(), [7, 7, 7])


def test_tf_paired_chunks_use_full_stride_cadence_and_absolute_history_when_tensorflow_is_available():
    tf = pytest.importorskip("tensorflow")
    actions = tf.constant(np.arange(14, dtype=np.float32)[:, None])
    traj = {
        "actions": actions,
        "observation": {"state": actions},
        "prompt": tf.constant([f"p{i}" for i in range(14)]),
        "episode_id": tf.constant(3, tf.int32),
        "frame_index": tf.range(14, dtype=tf.int32),
    }

    result = build_tf_paired_chunks(traj, horizon=10, stride=2, history_length=4, tf=tf)

    np.testing.assert_array_equal(result["previous"]["actions"].numpy()[:, :, 0], [np.arange(10), np.arange(2, 12)])
    np.testing.assert_array_equal(result["current"]["actions"].numpy()[:, :, 0], [np.arange(2, 12), np.arange(4, 14)])
    np.testing.assert_array_equal(result["current"]["action_history"].numpy()[0, :, 0], [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_array_equal(result["current"]["action_history_mask"].numpy()[0], [False, False, True, True])
    np.testing.assert_array_equal(result["current"]["frame_index"].numpy(), [2, 4])
