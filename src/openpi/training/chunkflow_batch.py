"""Episode-aware batch primitives for ChunkFlow training."""

from collections.abc import Callable, Mapping, Sequence
import copy
from typing import SupportsIndex

from flax import struct
import numpy as np


@struct.dataclass
class PairedChunkBatch:
    """Adjacent action chunks kept together through loading and shuffling."""

    previous_observation: object
    previous_actions: object
    observation: object
    actions: object
    step: object


class PairedTransformedDataset:
    """Random-access dataset whose adjacent chunks remain one shuffle unit.

    ``pre_transforms`` canonicalize raw dataset keys. History is then extracted
    in raw action coordinates and attached to each paired record before
    ``transforms`` apply action deltas, normalization, and model-width padding.
    """

    _OPTIONAL_FIELDS = (
        "action_history",
        "action_history_mask",
        "rewards",
        "discounts",
        "executed_actions",
    )

    def __init__(
        self,
        dataset,
        *,
        episode_ids: np.ndarray,
        frame_indices: np.ndarray,
        stride: int,
        history_length: int,
        action_horizon: int | None = None,
        pre_transforms: Sequence[Callable[[Mapping], Mapping]] = (),
        transforms: Sequence[Callable[[Mapping], Mapping]] = (),
    ):
        if history_length < 0:
            raise ValueError("history_length must be non-negative")
        if action_horizon is not None and history_length > action_horizon:
            raise ValueError("history_length must satisfy 0 <= history_length <= action_horizon")
        if len(dataset) != len(np.asarray(episode_ids)) or len(dataset) != len(np.asarray(frame_indices)):
            raise ValueError("dataset and episode/frame metadata must have the same length")

        self._dataset = dataset
        self._pair_indices = tuple(
            paired_start_indices(
                episode_ids,
                frame_indices,
                stride=stride,
                action_horizon=action_horizon,
            )
        )
        self._record_keys = tuple(
            (int(episode), int(frame))
            for episode, frame in zip(np.asarray(episode_ids), np.asarray(frame_indices), strict=True)
        )
        self._record_lookup = {key: index for index, key in enumerate(self._record_keys)}
        self._stride = stride
        self._history_length = history_length
        self._pre_transforms = tuple(pre_transforms)
        self._transforms = tuple(transforms)

    @property
    def pair_indices(self) -> tuple[tuple[int, int], ...]:
        return self._pair_indices

    def __getitem__(self, index: SupportsIndex) -> dict:
        previous_index, current_index = self._pair_indices[index.__index__()]
        previous_raw = copy.deepcopy(self._dataset[previous_index])
        current_raw = copy.deepcopy(self._dataset[current_index])
        previous_optional = {key: previous_raw[key] for key in self._OPTIONAL_FIELDS if key in previous_raw}
        current_optional = {key: current_raw[key] for key in self._OPTIONAL_FIELDS if key in current_raw}
        previous = dict(self._apply(previous_raw, self._pre_transforms))
        current = dict(self._apply(current_raw, self._pre_transforms))
        previous.update(previous_optional)
        current.update(current_optional)

        if "actions" not in previous or "actions" not in current:
            raise KeyError("paired records must contain actions after pre_transforms")
        previous_actions_for_history = self._validate_pair_actions(previous["actions"])
        current_actions_for_history = self._validate_pair_actions(current["actions"])
        action_cache = {
            self._record_keys[previous_index]: previous_actions_for_history.copy(),
            self._record_keys[current_index]: current_actions_for_history.copy(),
        }
        previous_history, previous_history_mask = self._history_before(
            previous_index, previous_actions_for_history, action_cache
        )
        current_history, current_history_mask = self._history_before(
            current_index, current_actions_for_history, action_cache
        )
        previous = dict(previous)
        previous["action_history"] = previous_history
        previous["action_history_mask"] = previous_history_mask
        current = dict(current)
        current["action_history"] = current_history
        current["action_history_mask"] = current_history_mask

        previous = self._apply(previous, self._transforms)
        current = self._apply(current, self._transforms)
        previous_observation, previous_actions = self._split_actions(previous)
        observation, actions = self._split_actions(current)
        return {
            "previous_observation": previous_observation,
            "previous_actions": previous_actions,
            "observation": observation,
            "actions": actions,
            # The train step is supplied by the optimizer loop before loss
            # evaluation; zero keeps collation shape-stable at loader time.
            "step": np.int32(0),
        }

    def __len__(self) -> int:
        return len(self._pair_indices)

    @staticmethod
    def _apply(data: Mapping, transforms: Sequence[Callable[[Mapping], Mapping]]) -> Mapping:
        for transform in transforms:
            data = transform(data)
        return data

    @staticmethod
    def _split_actions(data: Mapping) -> tuple[dict, object]:
        if "actions" not in data:
            raise KeyError("transformed paired records must contain actions")
        return {key: value for key, value in data.items() if key != "actions"}, data["actions"]

    def _validate_pair_actions(self, actions: object) -> np.ndarray:
        actions = np.asarray(actions)
        if actions.ndim != 2:
            raise ValueError("paired record actions must be rank 2 [horizon, action_dim]")
        if self._history_length > actions.shape[0]:
            raise ValueError("history_length must not exceed the paired record action horizon")
        if self._stride > actions.shape[0]:
            raise ValueError("stride must not exceed the paired record action horizon")
        return actions

    def _history_before(
        self,
        record_index: int,
        actions: np.ndarray,
        action_cache: dict[tuple[int, int], np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        history = np.zeros((self._history_length, actions.shape[-1]), dtype=actions.dtype)
        history_mask = np.zeros((self._history_length,), dtype=bool)
        episode, frame = self._record_keys[record_index]
        for position in range(self._history_length):
            history_frame = frame - self._history_length + position
            source_key = (episode, history_frame)
            source_index = self._record_lookup.get(source_key)
            if source_index is None:
                continue
            source_actions = action_cache.get(source_key)
            if source_actions is None:
                source = self._apply(copy.deepcopy(self._dataset[source_index]), self._pre_transforms)
                if "actions" not in source:
                    raise KeyError("history source record must contain actions after pre_transforms")
                source_actions = np.asarray(source["actions"]).copy()
                action_cache[source_key] = source_actions
            if source_actions.ndim != 2 or source_actions.shape[-1] != actions.shape[-1]:
                raise ValueError("history source actions must match the paired action shape")
            history[position] = source_actions[0]
            history_mask[position] = True
        return history, history_mask


def paired_start_indices(
    episode_ids: np.ndarray,
    frame_indices: np.ndarray,
    *,
    stride: int,
    action_horizon: int | None = None,
) -> list[tuple[int, int]]:
    """Return ``(previous, current)`` records separated by ``stride`` frames.

    Pairing is based on episode/frame identity rather than array adjacency, so
    samples remain correct even if the source records are interleaved. Returned
    pairs follow current-record order and never span episodes.
    """

    if stride <= 0:
        raise ValueError("stride must be positive")
    if action_horizon is not None and not 0 < stride <= action_horizon:
        raise ValueError("action_horizon must satisfy 0 < stride <= action_horizon")

    episode_ids = np.asarray(episode_ids)
    frame_indices = np.asarray(frame_indices)
    if episode_ids.ndim != 1 or frame_indices.ndim != 1:
        raise ValueError("episode_ids and frame_indices must be rank 1")
    if episode_ids.shape[0] != frame_indices.shape[0]:
        raise ValueError("episode_ids and frame_indices must have the same length")

    lookup: dict[tuple[int, int], int] = {}
    records: list[tuple[int, int]] = []
    episode_starts: dict[int, int] = {}
    for index, (episode, frame) in enumerate(zip(episode_ids, frame_indices, strict=True)):
        key = (int(episode), int(frame))
        if key in lookup:
            raise ValueError(f"duplicate episode/frame key: {key}")
        lookup[key] = index
        records.append(key)
        episode_starts[key[0]] = min(episode_starts.get(key[0], key[1]), key[1])

    pairs = []
    for current, (episode, frame) in enumerate(records):
        previous_frame = frame - stride
        previous = lookup.get((episode, previous_frame))
        cadence_aligned = (previous_frame - episode_starts[episode]) % stride == 0
        full_windows = action_horizon is None or all(
            (episode, required_frame) in lookup for required_frame in range(previous_frame, frame + action_horizon)
        )
        if previous is not None and cadence_aligned and full_windows:
            pairs.append((previous, current))
    return pairs


def history_from_previous_chunk(
    previous_actions: np.ndarray,
    *,
    stride: int,
    history_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract demonstrated actions committed before the current chunk.

    For adjacent chunks whose starts are separated by ``stride``, positions
    ``stride - history_length : stride`` of the previous action target are the
    actions immediately preceding the current observation.
    """

    previous_actions = np.asarray(previous_actions)
    if previous_actions.ndim != 2:
        raise ValueError("previous_actions must be a rank 2 [horizon, action_dim] array")
    if not 0 <= history_length <= stride <= previous_actions.shape[0]:
        raise ValueError("history_length must satisfy 0 <= history_length <= stride <= horizon")

    values = previous_actions[stride - history_length : stride].copy()
    return values, np.ones((history_length,), dtype=bool)
