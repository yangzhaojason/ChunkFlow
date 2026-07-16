from __future__ import annotations

import numpy as np
import pytest

from openpi.policies.rtc_policy import RTCConfig, RTCPolicy


class FakePolicy:
    def __init__(self, chunks):
        self._chunks = iter(chunks)

    def infer(self, observation):
        del observation
        return {"actions": next(self._chunks)}


def test_overlap_uses_predictions_aligned_after_stride_without_mutating_base_chunks():
    chunks = [
        np.arange(6, dtype=np.float32)[:, None],
        np.arange(10, 16, dtype=np.float32)[:, None],
    ]
    originals = [chunk.copy() for chunk in chunks]
    policy = RTCPolicy(
        FakePolicy(chunks),
        RTCConfig(overlap_size=2, replan_interval=2, blending_method="linear"),
    )

    outputs = [policy.infer({})["actions"] for _ in range(4)]

    np.testing.assert_allclose(outputs, [[0], [1], [2], [11]])
    transition = policy.get_metrics()["chunk_transitions"][0]
    np.testing.assert_allclose(transition["prev_tail"], [[2], [3]])
    np.testing.assert_allclose(transition["new_head"], [[10], [11]])
    np.testing.assert_allclose(transition["blended"], [[2], [11]])
    assert policy.get_metrics()["boundary_consistency"] == [pytest.approx(8.0)]
    for chunk, original in zip(chunks, originals, strict=True):
        np.testing.assert_array_equal(chunk, original)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"overlap_size": -1},
        {"overlap_size": 1.5},
        {"replan_interval": 0},
        {"replan_interval": True},
        {"blending_method": "cubic"},
    ],
)
def test_invalid_rtc_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        RTCConfig(**kwargs)


def test_policy_chunk_must_be_rank_two():
    policy = RTCPolicy(FakePolicy([np.zeros((4,), dtype=np.float32)]), RTCConfig())

    with pytest.raises(ValueError, match=r"shape \[H, D\]"):
        policy.infer({})


def test_policy_chunk_must_cover_replan_and_overlap():
    policy = RTCPolicy(
        FakePolicy([np.zeros((3, 1), dtype=np.float32)]),
        RTCConfig(replan_interval=2, overlap_size=2),
    )

    with pytest.raises(ValueError, match=r"replan_interval \+ overlap_size"):
        policy.infer({})


def test_action_width_must_remain_constant_across_replans():
    policy = RTCPolicy(
        FakePolicy(
            [
                np.zeros((4, 1), dtype=np.float32),
                np.zeros((4, 2), dtype=np.float32),
            ]
        ),
        RTCConfig(replan_interval=2, overlap_size=2),
    )
    policy.infer({})
    policy.infer({})

    with pytest.raises(ValueError, match="action dimension changed"):
        policy.infer({})


def test_reset_allows_a_new_action_width():
    policy = RTCPolicy(
        FakePolicy(
            [
                np.zeros((4, 1), dtype=np.float32),
                np.zeros((4, 2), dtype=np.float32),
            ]
        ),
        RTCConfig(replan_interval=2, overlap_size=2),
    )
    policy.infer({})

    policy.reset()

    assert policy.infer({})["actions"].shape == (2,)


def test_integer_chunks_are_promoted_before_fractional_blending():
    policy = RTCPolicy(
        FakePolicy(
            [
                np.arange(6, dtype=np.int64)[:, None],
                np.full((6, 1), 10, dtype=np.int64),
            ]
        ),
        RTCConfig(replan_interval=2, overlap_size=3, blending_method="linear"),
    )

    outputs = [policy.infer({})["actions"] for _ in range(4)]

    np.testing.assert_allclose(outputs, [[0.0], [1.0], [2.0], [6.5]])
    assert all(np.issubdtype(action.dtype, np.floating) for action in outputs)


def test_action_dimension_must_be_nonzero():
    policy = RTCPolicy(
        FakePolicy([np.zeros((3, 0), dtype=np.float32)]),
        RTCConfig(replan_interval=2, overlap_size=1),
    )

    with pytest.raises(ValueError, match="action dimension must be positive"):
        policy.infer({})


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_policy_actions_must_be_finite(invalid_value):
    chunk = np.zeros((4, 1), dtype=np.float32)
    chunk[0, 0] = invalid_value
    policy = RTCPolicy(
        FakePolicy([chunk]),
        RTCConfig(replan_interval=2, overlap_size=2),
    )

    with pytest.raises(ValueError, match="finite"):
        policy.infer({})


def test_get_metrics_returns_deep_copies():
    policy = RTCPolicy(
        FakePolicy([np.arange(4, dtype=np.float32)[:, None]]),
        RTCConfig(replan_interval=2, overlap_size=2),
    )
    policy.infer({})

    exported = policy.get_metrics()
    exported["action_history"][0][0] = 99
    exported["chunk_history"][0][0, 0] = 99

    fresh = policy.get_metrics()
    np.testing.assert_array_equal(fresh["action_history"][0], [0])
    np.testing.assert_array_equal(fresh["chunk_history"][0][:2], [[0], [1]])


def test_single_step_overlap_prefers_previous_aligned_prediction():
    policy = RTCPolicy(
        FakePolicy(
            [
                np.arange(3, dtype=np.float32)[:, None],
                np.arange(10, 13, dtype=np.float32)[:, None],
            ]
        ),
        RTCConfig(replan_interval=2, overlap_size=1, blending_method="linear"),
    )

    outputs = [policy.infer({})["actions"] for _ in range(3)]

    np.testing.assert_allclose(outputs, [[0], [1], [2]])
