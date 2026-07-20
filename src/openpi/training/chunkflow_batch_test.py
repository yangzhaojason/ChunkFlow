import numpy as np
import pytest

from openpi.shared import normalize as _normalize
from openpi.training import chunkflow_batch as _chunkflow_batch
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


class _CountingDataset(_ListDataset):
    def __init__(self, records):
        super().__init__(records)
        self.read_counts = np.zeros((len(records),), dtype=np.int64)

    def __getitem__(self, index):
        self.read_counts[index] += 1
        return super().__getitem__(index)


def _transition_records():
    return [
        {
            "state": np.array([float(frame)], dtype=np.float32),
            "actions": np.array([[900.0 + frame]], dtype=np.float32),
            "executed": np.array([100.0 + frame], dtype=np.float32),
            "reward": np.float32(frame == 2),
            "cont": np.float32(frame < 2),
        }
        for frame in range(3)
    ]


def _step_transition_dataset(records=None, *, history_length=2, transforms=()):
    records = _transition_records() if records is None else records
    return _chunkflow_batch.StepTransitionDataset(
        _ListDataset(records),
        episode_ids=np.zeros(len(records), dtype=np.int64),
        frame_indices=np.arange(len(records), dtype=np.int64),
        history_length=history_length,
        executed_action_key="executed",
        reward_key="reward",
        continuation_key="cont",
        transforms=transforms,
    )


def test_step_transition_dataset_emits_every_frame_and_terminal_without_crossing_episodes():
    records = [
        {"state": np.array([0.0]), "actions": np.array([[10.0]]), "reward": 0.0, "cont": 1.0},
        {"state": np.array([7.0]), "actions": np.array([[70.0]]), "reward": 0.0, "cont": 1.0},
        {"state": np.array([1.0]), "actions": np.array([[11.0]]), "reward": 1.0, "cont": 0.0},
        {"state": np.array([8.0]), "actions": np.array([[71.0]]), "reward": 1.0, "cont": 0.0},
    ]
    dataset = _chunkflow_batch.StepTransitionDataset(
        _ListDataset(records),
        episode_ids=np.array([0, 1, 0, 1]),
        frame_indices=np.array([0, 0, 1, 1]),
        history_length=2,
        executed_action_key="actions",
        reward_key="reward",
        continuation_key="cont",
    )

    assert len(dataset) == 4
    first = dataset[0]
    other_episode = dataset[1]
    terminal = dataset[2]
    assert first["episode_id"] == 0
    assert first["frame_index"] == 0
    assert first["next_observation"]["state"].item() == 1.0
    assert other_episode["next_observation"]["state"].item() == 8.0
    assert terminal["continuation"] == 0.0
    assert terminal["next_observation"]["state"].item() == terminal["observation"]["state"].item()


def test_step_transition_histories_use_only_explicit_executed_actions():
    dataset = _chunkflow_batch.StepTransitionDataset(
        _ListDataset(_transition_records()),
        episode_ids=np.zeros(3, dtype=np.int64),
        frame_indices=np.arange(3),
        history_length=2,
        executed_action_key="executed",
        reward_key="reward",
        continuation_key="cont",
    )

    second = dataset[1]

    np.testing.assert_array_equal(second["observation"]["action_history_mask"], [False, True])
    np.testing.assert_array_equal(second["observation"]["action_history"], [[0.0], [100.0]])
    np.testing.assert_array_equal(second["next_observation"]["action_history_mask"], [True, True])
    np.testing.assert_array_equal(second["next_observation"]["action_history"], [[100.0], [101.0]])
    np.testing.assert_array_equal(second["executed_action"], [101.0])


def test_step_transition_applies_one_action_transform_chain_to_action_and_histories():
    records = [
        {"state": np.array([1.0]), "executed": np.array([2.0]), "reward": 0.0, "cont": 1.0},
        {"state": np.array([3.0]), "executed": np.array([5.0]), "reward": 0.0, "cont": 0.0},
    ]
    dataset = _chunkflow_batch.StepTransitionDataset(
        _ListDataset(records),
        episode_ids=np.zeros(2, dtype=np.int64),
        frame_indices=np.arange(2),
        history_length=1,
        executed_action_key="executed",
        reward_key="reward",
        continuation_key="cont",
        transforms=(
            _transforms.DeltaActions(mask=[True]),
            _transforms.PadStatesAndActions(model_action_dim=3),
        ),
    )

    second = dataset[1]

    np.testing.assert_array_equal(second["executed_action"], [2.0, 0.0, 0.0])
    np.testing.assert_array_equal(second["observation"]["action_history"], [[-1.0, 0.0, 0.0]])


@pytest.mark.parametrize("missing", ["executed", "reward", "cont"])
def test_step_transition_rejects_missing_required_field(missing):
    records = _transition_records()
    records[0].pop(missing)
    dataset = _step_transition_dataset(records)

    with pytest.raises((KeyError, ValueError), match=missing):
        dataset[0]


def test_step_transition_rejects_duplicate_episode_frame_keys_at_construction():
    with pytest.raises(ValueError, match="duplicate"):
        _chunkflow_batch.StepTransitionDataset(
            _ListDataset(_transition_records()[:2]),
            episode_ids=np.array([4, 4]),
            frame_indices=np.array([7, 7]),
            history_length=0,
            executed_action_key="executed",
            reward_key="reward",
            continuation_key="cont",
        )


def test_step_transition_requires_nonnegative_history_length():
    with pytest.raises(ValueError, match="history_length"):
        _step_transition_dataset(history_length=-1)


@pytest.mark.parametrize(
    ("executed", "expected"),
    [
        (np.array([3.0, 4.0]), [3.0, 4.0]),
        (np.array([[3.0, 4.0], [8.0, 9.0]]), [3.0, 4.0]),
    ],
)
def test_step_transition_accepts_vector_or_nonempty_sequence_executed_action(executed, expected):
    records = _transition_records()
    records[0]["executed"] = executed
    records[1]["executed"] = np.asarray(expected, dtype=np.float64)
    dataset = _step_transition_dataset(records, history_length=0)

    np.testing.assert_array_equal(dataset[0]["executed_action"], expected)


@pytest.mark.parametrize(
    "executed",
    [
        np.array([], dtype=np.float32),
        np.empty((0, 1), dtype=np.float32),
        np.array(1.0),
        np.zeros((1, 1, 1), dtype=np.float32),
        np.array([np.nan], dtype=np.float32),
        np.array([1.0 + 1.0j]),
        np.array([1], dtype=np.int32),
    ],
)
def test_step_transition_rejects_invalid_executed_action(executed):
    records = _transition_records()
    records[0]["executed"] = executed
    dataset = _step_transition_dataset(records, history_length=0)

    with pytest.raises(ValueError, match="executed"):
        dataset[0]


def test_step_transition_accepts_scalar_or_one_element_reward_and_continuation():
    records = _transition_records()
    records[0]["reward"] = np.array([2.5])
    records[0]["cont"] = np.array(0.25)

    transition = _step_transition_dataset(records)[0]

    assert transition["reward"] == 2.5
    assert transition["continuation"] == 0.25


def test_step_transition_normalizes_float64_scalar_fields_to_float32():
    records = _transition_records()
    records[0]["reward"] = np.float64(0.25)
    records[0]["cont"] = np.array([0.5], dtype=np.float64)

    transition = _step_transition_dataset(records)[0]

    assert isinstance(transition["reward"], np.float32)
    assert isinstance(transition["continuation"], np.float32)
    assert transition["reward"].dtype == np.float32
    assert transition["continuation"].dtype == np.float32
    assert transition["reward"] == np.float32(0.25)
    assert transition["continuation"] == np.float32(0.5)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reward", np.array([1.0, 2.0])),
        ("reward", np.nan),
        ("cont", np.array([0.0, 1.0])),
        ("cont", np.nan),
        ("cont", np.inf),
        ("cont", -0.01),
        ("cont", 1.01),
    ],
)
def test_step_transition_rejects_invalid_scalar_fields(field, value):
    records = _transition_records()
    records[0][field] = value
    dataset = _step_transition_dataset(records)

    with pytest.raises(ValueError, match=field):
        dataset[0]


def test_step_transition_rejects_missing_next_frame_with_nonzero_continuation():
    records = [{"state": np.array([0.0]), "executed": np.array([1.0]), "reward": 0.0, "cont": 1.0}]

    with pytest.raises(ValueError, match="continuation"):
        _step_transition_dataset(records, history_length=0)[0]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("executed_actions", np.zeros((1, 1, 1)), "executed"),
        ("action_history", np.zeros((1, 1, 1)), "history"),
        ("action_history", np.zeros((1, 2)), "shape"),
        ("executed_actions", np.array([[np.inf]]), "finite"),
        ("state", np.array([np.nan]), "finite"),
    ],
)
def test_step_transition_validates_transformed_shapes_and_floating_leaves(field, value, message):
    def corrupt(data):
        return {**data, field: value}

    dataset = _step_transition_dataset(history_length=1, transforms=(corrupt,))

    with pytest.raises(ValueError, match=message):
        dataset[0]


def test_step_transition_terminal_history_shifts_and_appends_current_action():
    terminal = _step_transition_dataset()[2]

    np.testing.assert_array_equal(terminal["observation"]["action_history"], [[100.0], [101.0]])
    np.testing.assert_array_equal(terminal["next_observation"]["action_history"], [[101.0], [102.0]])
    np.testing.assert_array_equal(terminal["next_observation"]["action_history_mask"], [True, True])


def test_step_transition_pretransforms_canonicalize_without_mutating_source_records():
    records = [
        {
            "raw_state": np.array([1.0]),
            "raw_targets": np.array([[9.0]]),
            "raw_executed": np.array([2.0]),
            "reward": 0.0,
            "cont": 0.0,
        }
    ]
    original = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in records[0].items()}
    dataset = _chunkflow_batch.StepTransitionDataset(
        _ListDataset(records),
        episode_ids=np.array([0]),
        frame_indices=np.array([0]),
        history_length=1,
        executed_action_key="raw_executed",
        reward_key="reward",
        continuation_key="cont",
        pre_transforms=(
            _transforms.RepackTransform({"state": "raw_state", "actions": "raw_targets"}),
        ),
    )

    transition = dataset[0]

    np.testing.assert_array_equal(transition["observation"]["state"], [1.0])
    assert "raw_executed" not in transition["observation"]
    for key, value in original.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(records[0][key], value)
        else:
            assert records[0][key] == value


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


@pytest.mark.parametrize(
    ("horizons", "stride", "history_length", "message"),
    [
        ((4, 6), 5, 4, r"stride.*horizon"),
        ((6, 4), 5, 4, r"stride.*horizon"),
        ((4, 6), 2, 5, r"history_length.*horizon"),
        ((6, 4), 2, 5, r"history_length.*horizon"),
    ],
)
def test_paired_dataset_validates_runtime_action_horizon_for_each_pair_record(
    horizons, stride, history_length, message
):
    records = [
        {
            "state": np.array([frame], dtype=np.float32),
            "actions": np.arange(horizon, dtype=np.float32)[:, None],
        }
        for frame, horizon in zip((0, stride), horizons, strict=True)
    ]
    dataset = PairedTransformedDataset(
        _ListDataset(records),
        episode_ids=np.zeros(2, dtype=np.int64),
        frame_indices=np.array([0, stride]),
        stride=stride,
        history_length=history_length,
    )

    with pytest.raises(ValueError, match=message):
        dataset[0]


def test_paired_dataset_rejects_mismatched_pair_horizons_with_zero_history():
    records = [
        {"state": np.array([0.0]), "actions": np.zeros((4, 1), dtype=np.float32)},
        {"state": np.array([2.0]), "actions": np.zeros((5, 1), dtype=np.float32)},
    ]
    dataset = PairedTransformedDataset(
        _ListDataset(records),
        episode_ids=np.zeros(2, dtype=np.int64),
        frame_indices=np.array([0, 2]),
        stride=2,
        history_length=0,
    )

    with pytest.raises(ValueError, match="identical shape"):
        dataset[0]


def test_paired_dataset_rejects_mismatched_pair_action_dims_with_zero_history():
    records = [
        {"state": np.array([0.0]), "actions": np.zeros((4, 1), dtype=np.float32)},
        {"state": np.array([2.0]), "actions": np.zeros((4, 2), dtype=np.float32)},
    ]
    dataset = PairedTransformedDataset(
        _ListDataset(records),
        episode_ids=np.zeros(2, dtype=np.int64),
        frame_indices=np.array([0, 2]),
        stride=2,
        history_length=0,
    )

    with pytest.raises(ValueError, match="identical shape"):
        dataset[0]


def test_paired_dataset_rejects_mismatched_pair_dtypes_with_zero_history():
    records = [
        {"state": np.array([0.0]), "actions": np.zeros((4, 1), dtype=np.float32)},
        {"state": np.array([2.0]), "actions": np.zeros((4, 1), dtype=np.float64)},
    ]
    dataset = PairedTransformedDataset(
        _ListDataset(records),
        episode_ids=np.zeros(2, dtype=np.int64),
        frame_indices=np.array([0, 2]),
        stride=2,
        history_length=0,
    )

    with pytest.raises(ValueError, match="identical dtype"):
        dataset[0]


@pytest.mark.parametrize(
    ("source_actions", "message"),
    [
        (np.zeros((4, 1), dtype=np.float64), r"history source.*dtype"),
        (np.zeros((3, 1), dtype=np.float32), r"history source.*shape"),
    ],
)
def test_paired_dataset_rejects_incompatible_history_source_chunks(source_actions, message):
    records = [
        {"state": np.array([float(frame)]), "actions": np.zeros((4, 1), dtype=np.float32)}
        for frame in range(5)
    ]
    records[1]["actions"] = source_actions
    dataset = PairedTransformedDataset(
        _ListDataset(records),
        episode_ids=np.zeros(5, dtype=np.int64),
        frame_indices=np.arange(5),
        stride=2,
        history_length=1,
    )

    with pytest.raises(ValueError, match=message):
        dataset[1]


def test_paired_dataset_caches_each_canonical_history_source_per_item():
    records = [
        {
            "frame": frame,
            "state": np.array([frame], dtype=np.float32),
            "actions": np.arange(frame, frame + 10, dtype=np.float32)[:, None],
        }
        for frame in range(7)
    ]
    source = _CountingDataset(records)
    pre_transform_counts = np.zeros((len(records),), dtype=np.int64)

    def count_pre_transform(record):
        pre_transform_counts[record["frame"]] += 1
        return record

    dataset = PairedTransformedDataset(
        source,
        episode_ids=np.zeros(len(records), dtype=np.int64),
        frame_indices=np.arange(len(records)),
        stride=2,
        history_length=4,
        pre_transforms=[count_pre_transform],
    )

    item = dataset[2]

    np.testing.assert_array_equal(item["previous_observation"]["action_history"][:, 0], [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(item["observation"]["action_history"][:, 0], [2.0, 3.0, 4.0, 5.0])
    np.testing.assert_array_equal(source.read_counts, np.ones((len(records),), dtype=np.int64))
    np.testing.assert_array_equal(pre_transform_counts, np.ones((len(records),), dtype=np.int64))


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
