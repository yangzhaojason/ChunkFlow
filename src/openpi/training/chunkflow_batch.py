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


@struct.dataclass
class StepTransitionBatch:
    observation: object
    executed_action: object
    reward: object
    continuation: object
    next_observation: object
    episode_id: object
    frame_index: object


@struct.dataclass
class ChunkFlowTrainBatch:
    supervised: PairedChunkBatch
    transition: StepTransitionBatch


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
        if previous_actions_for_history.shape != current_actions_for_history.shape:
            raise ValueError("paired previous/current actions must have identical shape [horizon, action_dim]")
        if previous_actions_for_history.dtype != current_actions_for_history.dtype:
            raise ValueError("paired previous/current actions must have identical dtype")
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
            if source_actions.ndim != 2 or source_actions.shape != actions.shape:
                raise ValueError("history source actions must match the paired action shape [horizon, action_dim]")
            if source_actions.dtype != actions.dtype:
                raise ValueError("history source actions must match the paired action dtype")
            history[position] = source_actions[0]
            history_mask[position] = True
        return history, history_mask


class StepTransitionDataset:
    """Every source frame represented as one strict, episode-local transition."""

    def __init__(
        self,
        dataset,
        *,
        episode_ids: np.ndarray,
        frame_indices: np.ndarray,
        history_length: int,
        executed_action_key: str,
        reward_key: str,
        continuation_key: str,
        pre_transforms: Sequence[Callable[[Mapping], Mapping]] = (),
        transforms: Sequence[Callable[[Mapping], Mapping]] = (),
    ):
        if history_length < 0:
            raise ValueError("history_length must be non-negative")

        episode_ids = np.asarray(episode_ids)
        frame_indices = np.asarray(frame_indices)
        if episode_ids.ndim != 1 or frame_indices.ndim != 1:
            raise ValueError("episode_ids and frame_indices must be rank 1")
        if episode_ids.shape[0] != frame_indices.shape[0]:
            raise ValueError("episode_ids and frame_indices must have the same length")
        if len(dataset) != episode_ids.shape[0]:
            raise ValueError("dataset and episode/frame metadata must have the same length")

        record_keys = tuple(
            (int(episode), int(frame))
            for episode, frame in zip(episode_ids, frame_indices, strict=True)
        )
        record_lookup: dict[tuple[int, int], int] = {}
        for index, key in enumerate(record_keys):
            if key in record_lookup:
                raise ValueError(f"duplicate episode/frame key: {key}")
            record_lookup[key] = index

        self._dataset = dataset
        self._record_keys = record_keys
        self._record_lookup = record_lookup
        self._history_length = history_length
        self._executed_action_key = executed_action_key
        self._reward_key = reward_key
        self._continuation_key = continuation_key
        self._pre_transforms = tuple(pre_transforms)
        self._transforms = tuple(transforms)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: SupportsIndex) -> dict:
        record_index = index.__index__()
        episode, frame = self._record_keys[record_index]
        current_raw = copy.deepcopy(self._dataset[record_index])
        executed = self._executed_action(current_raw)
        reward = self._scalar(current_raw, self._reward_key)
        continuation = self._scalar(current_raw, self._continuation_key)
        self._validate_reward_and_continuation(reward, continuation)
        history, history_mask = self._history_before(record_index, executed)

        next_index = self._record_lookup.get((episode, frame + 1))
        if next_index is None:
            if continuation != 0:
                raise ValueError(
                    f"continuation field {self._continuation_key!r} must be exactly 0 "
                    "when the next episode frame is missing"
                )
            next_raw = copy.deepcopy(current_raw)
            next_executed = executed.copy()
            if self._history_length > 0:
                next_history = np.concatenate((history[1:], executed[None]), axis=0)
                next_history_mask = np.concatenate((history_mask[1:], np.ones((1,), dtype=bool)))
            else:
                next_history = history.copy()
                next_history_mask = history_mask.copy()
        else:
            next_raw = copy.deepcopy(self._dataset[next_index])
            next_executed = self._executed_action(next_raw)
            next_history, next_history_mask = self._history_before(next_index, next_executed)

        observation, transformed_executed = self._transform_record(
            current_raw,
            executed=executed,
            history=history,
            history_mask=history_mask,
        )
        next_observation, transformed_next_executed = self._transform_record(
            next_raw,
            executed=next_executed,
            history=next_history,
            history_mask=next_history_mask,
        )
        if transformed_executed.shape != transformed_next_executed.shape:
            raise ValueError("current and next transformed executed action shapes must match")

        return {
            "observation": observation,
            "executed_action": transformed_executed,
            "reward": reward,
            "continuation": continuation,
            "next_observation": next_observation,
            "episode_id": np.int32(episode),
            "frame_index": np.int32(frame),
        }

    def _executed_action(self, record: Mapping) -> np.ndarray:
        if self._executed_action_key not in record:
            raise KeyError(f"missing required executed action field: {self._executed_action_key}")
        values = np.asarray(record[self._executed_action_key])
        if values.ndim == 1:
            if values.shape[0] == 0:
                raise ValueError(f"{self._executed_action_key} executed action must be nonempty")
            action = values
        elif values.ndim == 2:
            if values.shape[0] == 0 or values.shape[1] == 0:
                raise ValueError(f"{self._executed_action_key} executed action sequence must be nonempty")
            action = values[0]
        else:
            raise ValueError(
                f"{self._executed_action_key} executed action must have shape [A] or [T, A]"
            )
        self._require_real_floating(action, name=f"{self._executed_action_key} executed action")
        return np.array(action, copy=True)

    @staticmethod
    def _scalar(record: Mapping, key: str):
        if key not in record:
            raise KeyError(f"missing required scalar field: {key}")
        value = np.asarray(record[key])
        if value.ndim == 0:
            scalar = value.item()
        elif value.size == 1:
            scalar = value.reshape(-1)[0].item()
        else:
            raise ValueError(f"{key} must be a scalar or one-element array")
        if not np.issubdtype(value.dtype, np.number) or not np.isrealobj(value):
            raise ValueError(f"{key} must be a real numeric scalar")
        return scalar

    def _validate_reward_and_continuation(self, reward, continuation) -> None:
        if not np.isfinite(reward):
            raise ValueError(f"{self._reward_key} must be finite")
        if not np.isfinite(continuation):
            raise ValueError(f"{self._continuation_key} must be finite")
        if not 0 <= continuation <= 1:
            raise ValueError(f"{self._continuation_key} must be within [0, 1]")

    def _history_before(self, record_index: int, executed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        history = np.zeros((self._history_length, executed.shape[-1]), dtype=executed.dtype)
        history_mask = np.zeros((self._history_length,), dtype=bool)
        episode, frame = self._record_keys[record_index]
        for position in range(self._history_length):
            history_frame = frame - self._history_length + position
            source_index = self._record_lookup.get((episode, history_frame))
            if source_index is None:
                continue
            source = copy.deepcopy(self._dataset[source_index])
            source_executed = self._executed_action(source)
            if source_executed.shape != executed.shape:
                raise ValueError("history executed action shape must match the current executed action shape")
            history[position] = source_executed
            history_mask[position] = True
        return history, history_mask

    def _transform_record(
        self,
        raw: Mapping,
        *,
        executed: np.ndarray,
        history: np.ndarray,
        history_mask: np.ndarray,
    ) -> tuple[dict, np.ndarray]:
        canonical = dict(self._apply(copy.deepcopy(raw), self._pre_transforms))
        canonical["executed_actions"] = executed[None].copy()
        canonical["action_history"] = history.copy()
        canonical["action_history_mask"] = history_mask.copy()
        transformed = dict(self._apply(canonical, self._transforms))

        if "executed_actions" not in transformed:
            raise KeyError("transformed transition record must contain executed_actions")
        transformed_executed = np.asarray(transformed["executed_actions"])
        if (
            transformed_executed.ndim != 2
            or transformed_executed.shape[0] != 1
            or transformed_executed.shape[1] == 0
        ):
            raise ValueError("transformed executed action must have shape [1, action_dim]")
        self._require_real_floating(transformed_executed, name="transformed executed action")

        if "action_history" not in transformed or "action_history_mask" not in transformed:
            raise KeyError("transformed transition record must contain action history and mask")
        transformed_history = np.asarray(transformed["action_history"])
        transformed_history_mask = np.asarray(transformed["action_history_mask"])
        if transformed_history.ndim != 2 or transformed_history.shape[0] != self._history_length:
            raise ValueError("transformed action history must have shape [history_length, action_dim]")
        if transformed_history.shape[1] != transformed_executed.shape[1]:
            raise ValueError("transformed action history shape must match the transformed executed action shape")
        if transformed_history_mask.shape != (self._history_length,):
            raise ValueError("transformed action history mask must have shape [history_length]")
        if not np.issubdtype(transformed_history_mask.dtype, np.bool_):
            raise ValueError("transformed action history mask must be boolean")
        self._require_real_floating(transformed_history, name="transformed action history")

        observation = self._observation_only(transformed)
        self._require_floating_leaves_finite(observation)
        return observation, np.array(transformed_executed[0], copy=True)

    def _observation_only(self, transformed: Mapping) -> dict:
        removed = {
            "actions",
            "executed_actions",
            "rewards",
            "discounts",
            self._executed_action_key,
            self._reward_key,
            self._continuation_key,
        }
        return {key: value for key, value in transformed.items() if key not in removed}

    @staticmethod
    def _apply(data: Mapping, transforms: Sequence[Callable[[Mapping], Mapping]]) -> Mapping:
        for transform in transforms:
            data = transform(data)
        return data

    @staticmethod
    def _require_real_floating(value: object, *, name: str) -> None:
        array = np.asarray(value)
        try:
            floating = np.issubdtype(array.dtype, np.floating)
        except TypeError:
            floating = False
        if not floating or not np.isrealobj(array):
            raise ValueError(f"{name} must use a real floating dtype")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")

    @classmethod
    def _require_floating_leaves_finite(cls, tree: object) -> None:
        if isinstance(tree, Mapping):
            for value in tree.values():
                cls._require_floating_leaves_finite(value)
            return
        if isinstance(tree, (tuple, list)):
            for value in tree:
                cls._require_floating_leaves_finite(value)
            return
        try:
            array = np.asarray(tree)
            inexact = np.issubdtype(array.dtype, np.inexact)
        except (TypeError, ValueError):
            return
        if inexact and not np.all(np.isfinite(array)):
            raise ValueError("transformed transition floating leaves must contain only finite values")


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
