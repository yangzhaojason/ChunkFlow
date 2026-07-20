import numpy as np
import pytest

from openpi.shared import normalize as _normalize
from openpi.training.chunkflow_batch import PairedChunkBatch
from openpi.training.chunkflow_batch import PairedTransformedDataset
from openpi.training.chunkflow_batch import history_from_previous_chunk
from openpi.training.chunkflow_batch import paired_start_indices
import openpi.transforms as _transforms


def test_pairs_use_stride_and_never_cross_episode():
    episodes = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    frames = np.array([0, 1, 2, 3, 0, 1, 2, 3])

    assert paired_start_indices(episodes, frames, stride=2) == [(0, 2), (4, 6)]


def test_pairs_follow_frame_identity_instead_of_array_adjacency():
    episodes = np.array([2, 1, 2, 1])
    frames = np.array([3, 0, 1, 2])

    assert paired_start_indices(episodes, frames, stride=2) == [(2, 0), (1, 3)]


@pytest.mark.parametrize("stride", [0, -1])
def test_pairs_require_positive_stride(stride):
    with pytest.raises(ValueError, match="stride must be positive"):
        paired_start_indices(np.array([0]), np.array([0]), stride=stride)


def test_pairs_reject_mismatched_metadata_lengths():
    with pytest.raises(ValueError, match="same length"):
        paired_start_indices(np.array([0, 0]), np.array([0]), stride=1)


def test_pairs_reject_duplicate_episode_frame_keys():
    with pytest.raises(ValueError, match="duplicate"):
        paired_start_indices(np.array([0, 0]), np.array([1, 1]), stride=1)


def test_pairs_require_two_complete_action_windows_when_horizon_is_known():
    episodes = np.zeros((6,), dtype=np.int64)
    frames = np.arange(6)

    assert paired_start_indices(episodes, frames, stride=2, action_horizon=4) == [(0, 2)]


def test_history_maps_to_previous_committed_positions():
    previous = np.arange(8, dtype=np.float32).reshape(4, 2)

    values, mask = history_from_previous_chunk(previous, stride=3, history_length=2)

    np.testing.assert_array_equal(values, previous[1:3])
    np.testing.assert_array_equal(mask, [True, True])
    assert not np.shares_memory(values, previous)


def test_zero_length_history_preserves_action_shape():
    previous = np.arange(8, dtype=np.float32).reshape(4, 2)

    values, mask = history_from_previous_chunk(previous, stride=3, history_length=0)

    assert values.shape == (0, 2)
    assert mask.shape == (0,)
    assert mask.dtype == np.bool_


@pytest.mark.parametrize(
    ("stride", "history_length"),
    [(2, 3), (5, 2), (2, -1), (-1, 0)],
)
def test_invalid_history_window_is_rejected(stride, history_length):
    with pytest.raises(ValueError, match="history_length"):
        history_from_previous_chunk(np.zeros((4, 2)), stride=stride, history_length=history_length)


def test_history_requires_a_rank_two_action_chunk():
    with pytest.raises(ValueError, match="rank 2"):
        history_from_previous_chunk(np.zeros((4,)), stride=2, history_length=1)


def test_paired_chunk_batch_is_a_jax_pytree():
    batch = PairedChunkBatch(
        previous_observation={"state": np.array([0.0])},
        previous_actions=np.array([[1.0]]),
        observation={"state": np.array([2.0])},
        actions=np.array([[3.0]]),
        step=np.array(4),
    )

    assert batch.previous_observation["state"].item() == 0.0
    assert batch.step.item() == 4


class _ListDataset:
    def __init__(self, records):
        self.records = records

    def __getitem__(self, index):
        return self.records[index]

    def __len__(self):
        return len(self.records)


def test_paired_transformed_dataset_attaches_raw_history_before_action_transforms():
    records = [
        {
            "state": np.array([7.0], dtype=np.float32),
            "actions": np.array([[8.0], [9.0], [11.0], [13.0]], dtype=np.float32),
        },
        {
            "state": np.array([8.0], dtype=np.float32),
            "actions": np.array([[10.0], [12.0], [14.0], [16.0]], dtype=np.float32),
        },
        {
            "state": np.array([0.0], dtype=np.float32),
            "actions": np.array([[9.0], [0.0], [0.0], [0.0]], dtype=np.float32),
        },
        {
            "state": np.array([0.0], dtype=np.float32),
            "actions": np.array([[11.0], [0.0], [0.0], [0.0]], dtype=np.float32),
        },
    ]
    originals = [{key: value.copy() for key, value in record.items()} for record in records]
    stats = {
        "actions": _normalize.NormStats(
            mean=np.array([10.0], dtype=np.float32),
            std=np.array([2.0], dtype=np.float32),
        )
    }
    transforms = [
        _transforms.DeltaActions(mask=[True]),
        _transforms.Normalize(stats),
        _transforms.PadStatesAndActions(model_action_dim=3),
    ]
    dataset = PairedTransformedDataset(
        _ListDataset(records),
        episode_ids=np.array([4, 4, 4, 4]),
        frame_indices=np.array([0, 3, 1, 2]),
        stride=3,
        history_length=2,
        transforms=transforms,
    )

    item = dataset[0]
    batch = PairedChunkBatch(**item)

    assert dataset.pair_indices == ((0, 1),)
    np.testing.assert_allclose(
        batch.observation["action_history"],
        [[-4.5, 0.0, 0.0], [-3.5, 0.0, 0.0]],
        atol=1e-5,
    )
    np.testing.assert_array_equal(batch.observation["action_history_mask"], [True, True])
    np.testing.assert_allclose(
        batch.actions[:2],
        [[-4.0, 0.0, 0.0], [-3.0, 0.0, 0.0]],
        atol=1e-5,
    )
    assert "actions" not in batch.previous_observation
    assert "actions" not in batch.observation
    assert batch.previous_actions.shape == (4, 3)
    np.testing.assert_array_equal(batch.previous_observation["action_history_mask"], [False, False])
    np.testing.assert_array_equal(batch.previous_observation["action_history"], np.zeros((2, 3)))
    assert np.asarray(batch.step).dtype == np.int32
    for record, original in zip(records, originals, strict=True):
        for key in original:
            np.testing.assert_array_equal(record[key], original[key])


def test_paired_transformed_dataset_applies_pretransforms_before_history_lookup():
    records = [
        {"raw_state": np.array([0.0]), "raw_actions": np.arange(4.0)[:, None]},
        {"raw_state": np.array([1.0]), "raw_actions": np.arange(4.0, 8.0)[:, None]},
        {"raw_state": np.array([0.5]), "raw_actions": np.arange(1.0, 5.0)[:, None]},
    ]
    dataset = PairedTransformedDataset(
        _ListDataset(records),
        episode_ids=np.array([0, 0, 0]),
        frame_indices=np.array([0, 2, 1]),
        stride=2,
        history_length=1,
        pre_transforms=[_transforms.RepackTransform({"state": "raw_state", "actions": "raw_actions"})],
    )

    item = dataset[0]

    np.testing.assert_array_equal(item["observation"]["action_history"], [[1.0]])
    np.testing.assert_array_equal(item["observation"]["action_history_mask"], [True])


def test_previous_observation_history_uses_episode_frames_and_left_padding():
    records = [
        {
            "state": np.array([float(frame)]),
            "actions": np.arange(frame, frame + 4, dtype=np.float32)[:, None],
        }
        for frame in range(7)
    ]
    dataset = PairedTransformedDataset(
        _ListDataset(records),
        episode_ids=np.zeros((7,), dtype=np.int64),
        frame_indices=np.arange(7),
        stride=2,
        history_length=2,
    )

    episode_start_pair = dataset[0]
    later_pair = dataset[1]

    np.testing.assert_array_equal(episode_start_pair["previous_observation"]["action_history"], [[0.0], [0.0]])
    np.testing.assert_array_equal(episode_start_pair["previous_observation"]["action_history_mask"], [False, False])
    np.testing.assert_array_equal(later_pair["previous_observation"]["action_history"], [[0.0], [1.0]])
    np.testing.assert_array_equal(later_pair["previous_observation"]["action_history_mask"], [True, True])


def test_paired_dataset_history_can_span_multiple_strides():
    records = [
        {"state": np.array([frame], dtype=np.float32), "actions": np.arange(frame, frame + 10, dtype=np.float32)[:, None]}
        for frame in range(14)
    ]
    dataset = PairedTransformedDataset(
        _ListDataset(records),
        episode_ids=np.zeros(14, dtype=np.int64),
        frame_indices=np.arange(14),
        stride=2,
        history_length=4,
        action_horizon=10,
    )
    item = dataset[0]
    np.testing.assert_array_equal(item["observation"]["action_history"][:, 0], [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_array_equal(item["observation"]["action_history_mask"], [False, False, True, True])


def test_map_style_pairing_restores_optional_fields_after_repacking():
    records = [
        {
            "raw_state": np.array([0.0]),
            "raw_actions": np.arange(4, dtype=np.float32)[:, None],
            "rewards": np.array([0.0, 1.0]),
            "discounts": np.array([1.0, 0.0]),
            "executed_actions": np.array([[2.0], [3.0]]),
        },
        {
            "raw_state": np.array([2.0]),
            "raw_actions": np.arange(2, 6, dtype=np.float32)[:, None],
            "rewards": np.array([1.0, 0.0]),
            "discounts": np.array([1.0, 1.0]),
            "executed_actions": np.array([[4.0], [5.0]]),
        },
    ]
    dataset = PairedTransformedDataset(
        _ListDataset(records),
        episode_ids=np.array([0, 0]),
        frame_indices=np.array([0, 2]),
        stride=2,
        history_length=1,
        pre_transforms=[_transforms.RepackTransform({"state": "raw_state", "actions": "raw_actions"})],
    )

    item = dataset[0]

    for key in ("rewards", "discounts", "executed_actions"):
        np.testing.assert_array_equal(item["previous_observation"][key], records[0][key])
        np.testing.assert_array_equal(item["observation"][key], records[1][key])
