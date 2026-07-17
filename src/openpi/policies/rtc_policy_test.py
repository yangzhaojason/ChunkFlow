from __future__ import annotations

import numpy as np
import pytest

from openpi.policies.rtc_policy import RTCConfig
from openpi.policies.rtc_policy import RTCPolicy


class FakePolicy:
    def __init__(self, chunks):
        self._chunks = iter(chunks)

    def infer(self, observation):
        del observation
        return {"actions": next(self._chunks)}


class RecordingPolicy(FakePolicy):
    def __init__(self, chunks):
        chunks = [np.asarray(chunk) for chunk in chunks]
        super().__init__(chunks)
        self.action_dim = chunks[0].shape[-1]
        self.observations = []

    def infer(self, observation):
        self.observations.append(
            {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in observation.items()
            }
        )
        return super().infer(observation)


class ModelWidthRecordingPolicy(FakePolicy):
    class _Model:
        action_dim = 32

    def __init__(self, chunks):
        super().__init__([np.asarray(chunk) for chunk in chunks])
        self._model = self._Model()
        self.observations = []

    def infer(self, observation):
        self.observations.append(
            {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in observation.items()
            }
        )
        return super().infer(observation)


class ResultPolicy:
    def __init__(self, result):
        self._result = result

    def infer(self, observation):
        del observation
        return self._result


class _HistoryModelConfig:
    history_length = 3


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
        {"history_length": -1},
        {"history_length": 1.5},
        {"history_length": True},
        {"blending_method": "cubic"},
    ],
)
def test_invalid_rtc_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        RTCConfig(**kwargs)


def test_rtc_config_derives_and_validates_model_history_length():
    config = RTCConfig.from_model_config(
        _HistoryModelConfig(),
        overlap_size=2,
        replan_interval=2,
    )

    assert config.history_length == 3
    with pytest.raises(ValueError, match=r"history_length.*model"):
        RTCConfig(history_length=2).validate_model_config(_HistoryModelConfig())


@pytest.mark.parametrize("result", [None, [], {}])
def test_policy_result_must_be_a_mapping_with_actions(result):
    policy = RTCPolicy(ResultPolicy(result), RTCConfig())

    with pytest.raises(ValueError, match=r"mapping containing.*actions"):
        policy.infer({})


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


def test_action_transform_runs_after_blending_before_metrics():
    policy = RTCPolicy(
        FakePolicy(
            [
                np.ones((5, 1), dtype=np.float32),
                -np.ones((5, 1), dtype=np.float32),
            ]
        ),
        RTCConfig(replan_interval=2, overlap_size=3, blending_method="linear"),
        action_transform=lambda action: np.where(action > 0.0, 1.0, -1.0),
    )

    outputs = [policy.infer({})["actions"] for _ in range(4)]

    np.testing.assert_array_equal(outputs, [[1.0], [1.0], [1.0], [-1.0]])
    np.testing.assert_array_equal(policy.get_metrics()["action_history"], outputs)


def test_rtc_injects_zero_padded_history_before_the_first_policy_call():
    base = RecordingPolicy(
        [
            np.arange(4, dtype=np.float32)[:, None],
            np.arange(4, 8, dtype=np.float32)[:, None],
        ]
    )
    policy = RTCPolicy(
        base,
        RTCConfig(overlap_size=2, replan_interval=2, history_length=3),
    )

    policy.infer({"state": np.array([1.0], dtype=np.float32)})

    np.testing.assert_array_equal(
        base.observations[0]["action_history"],
        np.zeros((3, 1), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        base.observations[0]["action_history_mask"],
        np.zeros((3,), dtype=bool),
    )

    policy.infer({})
    policy.infer({})

    np.testing.assert_array_equal(base.observations[1]["action_history"], [[0.0], [0.0], [1.0]])
    np.testing.assert_array_equal(base.observations[1]["action_history_mask"], [False, True, True])


def test_rtc_learns_environment_action_width_instead_of_using_model_padding_width():
    base = ModelWidthRecordingPolicy(
        [
            np.zeros((4, 7), dtype=np.float32),
            np.ones((4, 7), dtype=np.float32),
            np.full((4, 7), 2.0, dtype=np.float32),
        ]
    )
    policy = RTCPolicy(
        base,
        RTCConfig(overlap_size=2, replan_interval=2, history_length=2),
    )

    outputs = [policy.infer({})["actions"] for _ in range(3)]

    assert outputs[0].shape == (7,)
    assert "action_history" not in base.observations[0]
    assert base.observations[1]["action_history"].shape == (2, 7)
    np.testing.assert_array_equal(base.observations[1]["action_history"], outputs[:2])

    policy.reset()
    policy.infer({})

    np.testing.assert_array_equal(base.observations[2]["action_history"], np.zeros((2, 7)))
    np.testing.assert_array_equal(base.observations[2]["action_history_mask"], [False, False])


def test_rtc_history_contains_only_actions_actually_returned_to_environment():
    base = RecordingPolicy(
        [
            np.arange(6, dtype=np.float32)[:, None],
            np.arange(10, 16, dtype=np.float32)[:, None],
            np.arange(20, 26, dtype=np.float32)[:, None],
        ]
    )
    policy = RTCPolicy(
        base,
        RTCConfig(
            overlap_size=2,
            replan_interval=2,
            history_length=2,
            blending_method="linear",
        ),
        action_transform=lambda action: -action,
    )

    outputs = [policy.infer({})["actions"] for _ in range(5)]

    np.testing.assert_array_equal(base.observations[1]["action_history"], outputs[:2])
    np.testing.assert_array_equal(base.observations[1]["action_history_mask"], [True, True])
    # These are post-blending, post-transform actions from the second chunk,
    # not either policy's raw overlap predictions.
    np.testing.assert_array_equal(base.observations[2]["action_history"], outputs[2:4])
    np.testing.assert_array_equal(base.observations[2]["action_history_mask"], [True, True])


def test_reset_clears_executed_history():
    base = RecordingPolicy(
        [
            np.arange(4, dtype=np.float32)[:, None],
            np.arange(4, 8, dtype=np.float32)[:, None],
        ]
    )
    policy = RTCPolicy(
        base,
        RTCConfig(overlap_size=2, replan_interval=2, history_length=2),
    )
    policy.infer({})

    policy.reset()
    policy.infer({})

    assert policy.total_steps == 1
    np.testing.assert_array_equal(base.observations[1]["action_history"], [[0.0], [0.0]])
    np.testing.assert_array_equal(base.observations[1]["action_history_mask"], [False, False])


def test_zero_history_length_keeps_base_observation_unchanged():
    base = RecordingPolicy([np.arange(4, dtype=np.float32)[:, None]])
    observation = {"state": np.array([1.0], dtype=np.float32)}
    policy = RTCPolicy(
        base,
        RTCConfig(overlap_size=2, replan_interval=2, history_length=0),
    )

    policy.infer(observation)

    assert base.observations[0].keys() == observation.keys()
    assert "action_history" not in observation


def test_executed_history_is_available_when_metric_tracking_is_disabled():
    base = RecordingPolicy(
        [
            np.arange(4, dtype=np.float32)[:, None],
            np.arange(4, 8, dtype=np.float32)[:, None],
        ]
    )
    policy = RTCPolicy(
        base,
        RTCConfig(
            overlap_size=2,
            replan_interval=2,
            history_length=2,
            track_metrics=False,
        ),
    )

    outputs = [policy.infer({})["actions"] for _ in range(3)]

    assert policy.get_metrics() == {}
    np.testing.assert_array_equal(base.observations[1]["action_history"], outputs[:2])


def test_injected_history_uses_stable_float32_dtype_across_reset():
    base = RecordingPolicy(
        [
            np.arange(4, dtype=np.float64)[:, None],
            np.arange(4, 8, dtype=np.float64)[:, None],
            np.arange(8, 12, dtype=np.float64)[:, None],
        ]
    )
    policy = RTCPolicy(
        base,
        RTCConfig(overlap_size=2, replan_interval=2, history_length=2),
    )

    [policy.infer({}) for _ in range(3)]
    policy.reset()
    policy.infer({})

    assert all(observation["action_history"].dtype == np.float32 for observation in base.observations)


def test_executed_history_rejects_float32_overflow_before_replanning():
    base = RecordingPolicy(
        [
            np.full((2, 1), np.finfo(np.float64).max),
            np.zeros((2, 1), dtype=np.float64),
        ]
    )
    policy = RTCPolicy(
        base,
        RTCConfig(overlap_size=1, replan_interval=1, history_length=1),
    )
    policy.infer({})

    with pytest.raises(ValueError, match=r"finite.*float32"):
        policy.infer({})
