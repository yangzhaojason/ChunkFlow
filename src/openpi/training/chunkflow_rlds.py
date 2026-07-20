"""Episode-local RLDS assembly helpers for strict ChunkFlow training streams."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import numbers
from typing import Any

import numpy as np


@dataclasses.dataclass(frozen=True)
class EpisodeStepTransitions:
    """Aligned pure-array reference for every transition in one episode."""

    indices: np.ndarray
    next_indices: np.ndarray
    actions: np.ndarray
    reward: np.ndarray
    continuation: np.ndarray
    history: np.ndarray
    history_mask: np.ndarray
    next_history: np.ndarray
    next_history_mask: np.ndarray


def episode_step_transition_arrays(
    actions: np.ndarray,
    rewards: np.ndarray,
    continuation: np.ndarray,
    history_length: int,
) -> EpisodeStepTransitions:
    """Build strict step transitions without dropping the terminal frame.

    Input floating dtypes are preserved. Histories use the action dtype, while
    indices and masks use ``int64`` and ``bool`` respectively.
    """

    if isinstance(history_length, bool) or not isinstance(history_length, numbers.Integral):
        raise ValueError("history_length must be an integer")
    if history_length < 0:
        raise ValueError("history_length must be non-negative")

    actions = _real_floating_array(actions, name="actions", rank=2)
    rewards = _real_floating_array(rewards, name="rewards", rank=1)
    continuation = _real_floating_array(continuation, name="continuation", rank=1)
    if actions.shape[0] == 0 or actions.shape[1] == 0:
        raise ValueError("actions must be nonempty")
    trajectory_length = actions.shape[0]
    if rewards.shape[0] != trajectory_length:
        raise ValueError("rewards length must match actions")
    if continuation.shape[0] != trajectory_length:
        raise ValueError("continuation length must match actions")
    if np.any((continuation < 0) | (continuation > 1)):
        raise ValueError("continuation values must be within [0, 1]")
    if continuation[-1] != 0:
        raise ValueError("terminal continuation must be exactly 0")

    indices = np.arange(trajectory_length, dtype=np.int64)
    next_indices = np.minimum(indices + 1, trajectory_length - 1)
    history_offsets = np.arange(int(history_length), dtype=np.int64)
    history_indices = indices[:, None] - int(history_length) + history_offsets[None, :]
    history_mask = history_indices >= 0
    safe_history_indices = np.maximum(history_indices, 0)
    history = actions[safe_history_indices].copy()
    history = np.where(history_mask[..., None], history, np.zeros((), dtype=actions.dtype))

    next_history = history[next_indices].copy()
    next_history_mask = history_mask[next_indices].copy()
    if history_length:
        next_history[-1] = np.concatenate((history[-1, 1:], actions[-1:]), axis=0)
        next_history_mask[-1] = np.concatenate((history_mask[-1, 1:], np.ones((1,), dtype=bool)))

    return EpisodeStepTransitions(
        indices=indices,
        next_indices=next_indices,
        actions=actions.copy(),
        reward=rewards.copy(),
        continuation=continuation.copy(),
        history=history,
        history_mask=history_mask,
        next_history=next_history,
        next_history_mask=next_history_mask,
    )


def build_tf_step_transitions(
    traj: Mapping[str, Any],
    *,
    history_length: int,
    tf: Any,
) -> dict[str, Any]:
    """Build graph-safe step transitions from one canonical RLDS trajectory."""

    history_length = _static_integer(history_length, name="history_length", minimum=0)
    _require_keys(
        traj,
        ("actions", "rewards", "discounts", "observation", "prompt", "episode_id"),
    )

    actions = _validated_tf_actions(traj["actions"], tf=tf)
    trajectory_length = tf.shape(actions)[0]
    rewards = _validated_tf_vector(
        traj["rewards"],
        name="rewards",
        length=trajectory_length,
        tf=tf,
    )
    continuation = _validated_tf_vector(
        traj["discounts"],
        name="discounts",
        length=trajectory_length,
        tf=tf,
    )
    continuation_checks = (
        tf.debugging.assert_greater_equal(
            continuation,
            tf.cast(0, continuation.dtype),
            message="discounts must be within [0, 1]",
        ),
        tf.debugging.assert_less_equal(
            continuation,
            tf.cast(1, continuation.dtype),
            message="discounts must be within [0, 1]",
        ),
        tf.debugging.assert_equal(
            continuation[-1],
            tf.cast(0, continuation.dtype),
            message="terminal continuation must be exactly 0",
        ),
    )
    with tf.control_dependencies(continuation_checks):
        continuation = tf.identity(continuation)

    indices = tf.range(trajectory_length, dtype=tf.int32)
    next_indices = tf.minimum(indices + 1, trajectory_length - 1)
    history, history_mask = _tf_history(actions, indices, history_length=history_length, tf=tf)
    if history_length == 0:
        next_history = tf.identity(history)
        next_history_mask = tf.identity(history_mask)
    else:
        next_history = tf.concat((history[:, 1:], actions[:, tf.newaxis, :]), axis=1)
        next_history_mask = tf.concat(
            (history_mask[:, 1:], tf.ones((trajectory_length, 1), dtype=tf.bool)),
            axis=1,
        )

    current_observation = _tf_gather_aligned_tree(
        traj["observation"],
        indices,
        length=trajectory_length,
        name="observation",
        tf=tf,
    )
    next_observation = _tf_gather_aligned_tree(
        traj["observation"],
        next_indices,
        length=trajectory_length,
        name="observation",
        tf=tf,
    )
    current_prompt = _tf_gather_aligned_value(
        traj["prompt"],
        indices,
        length=trajectory_length,
        name="prompt",
        tf=tf,
    )
    next_prompt = _tf_gather_aligned_value(
        traj["prompt"],
        next_indices,
        length=trajectory_length,
        name="prompt",
        tf=tf,
    )
    episode_ids = _tf_aligned_or_scalar(
        traj["episode_id"],
        length=trajectory_length,
        name="episode_id",
        tf=tf,
    )
    if not episode_ids.dtype.is_integer:
        raise ValueError("episode_id must use an integer dtype")
    episode_ids = tf.cast(episode_ids, tf.int32)

    if "frame_index" in traj:
        frame_indices = _tf_aligned_or_scalar(
            traj["frame_index"],
            length=trajectory_length,
            name="frame_index",
            tf=tf,
        )
        if not frame_indices.dtype.is_integer:
            raise ValueError("frame_index must use an integer dtype")
        frame_indices = tf.cast(frame_indices, tf.int32)
    else:
        frame_indices = indices

    action_sequences = actions[:, tf.newaxis, :]
    reward_sequences = rewards[:, tf.newaxis]
    continuation_sequences = continuation[:, tf.newaxis]
    next_action_sequences = tf.gather(action_sequences, next_indices)
    next_reward_sequences = tf.gather(reward_sequences, next_indices)
    next_continuation_sequences = tf.gather(continuation_sequences, next_indices)

    current = {
        "actions": action_sequences,
        "executed_actions": action_sequences,
        "rewards": reward_sequences,
        "discounts": continuation_sequences,
        "observation": current_observation,
        "prompt": current_prompt,
        "action_history": history,
        "action_history_mask": history_mask,
    }
    next_record = {
        "actions": next_action_sequences,
        "executed_actions": next_action_sequences,
        "rewards": next_reward_sequences,
        "discounts": next_continuation_sequences,
        "observation": next_observation,
        "prompt": next_prompt,
        "action_history": next_history,
        "action_history_mask": next_history_mask,
    }
    for key in ("passes_filter", "adapter_mask"):
        if key in traj:
            current[key] = _tf_gather_aligned_value(
                traj[key],
                indices,
                length=trajectory_length,
                name=key,
                tf=tf,
            )
            next_record[key] = _tf_gather_aligned_value(
                traj[key],
                next_indices,
                length=trajectory_length,
                name=key,
                tf=tf,
            )
    return {
        "current": current,
        "next": next_record,
        "reward": rewards,
        "continuation": continuation,
        "episode_id": episode_ids,
        "frame_index": frame_indices,
    }


def build_tf_paired_chunks(
    traj: Mapping[str, Any],
    *,
    horizon: int,
    stride: int,
    history_length: int,
    tf: Any,
) -> dict[str, Any]:
    """Build graph-safe full previous/current chunks from one trajectory."""

    horizon = _static_integer(horizon, name="horizon", minimum=1)
    stride = _static_integer(stride, name="stride", minimum=1)
    history_length = _static_integer(history_length, name="history_length", minimum=0)
    if stride > horizon:
        raise ValueError("stride must satisfy 0 < stride <= horizon")
    if history_length > horizon:
        raise ValueError("history_length must satisfy 0 <= history_length <= horizon")
    _require_keys(traj, ("actions", "observation", "prompt"))

    actions = _validated_tf_actions(traj["actions"], tf=tf)
    trajectory_length = tf.shape(actions)[0]
    pair_count = tf.maximum(trajectory_length - stride - horizon + 1, 0)
    starts = tf.range(0, pair_count, delta=stride, dtype=tf.int32)
    offsets = tf.range(horizon, dtype=tf.int32)[tf.newaxis, :]
    previous_indices = starts[:, tf.newaxis] + offsets
    current_starts = starts + stride
    current_indices = current_starts[:, tf.newaxis] + offsets

    previous = _tf_paired_record(
        traj,
        starts=starts,
        chunk_indices=previous_indices,
        actions=actions,
        trajectory_length=trajectory_length,
        history_length=history_length,
        tf=tf,
    )
    current = _tf_paired_record(
        traj,
        starts=current_starts,
        chunk_indices=current_indices,
        actions=actions,
        trajectory_length=trajectory_length,
        history_length=history_length,
        tf=tf,
    )
    return {"previous": previous, "current": current}


def _static_integer(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _require_keys(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise KeyError(f"missing required canonical trajectory keys: {', '.join(missing)}")


def _validated_tf_actions(value: object, *, tf: Any) -> Any:
    actions = tf.convert_to_tensor(value)
    if not actions.dtype.is_floating:
        raise ValueError("actions must use a real floating dtype")
    rank_check = tf.debugging.assert_rank(actions, 2, message="actions must be rank 2 [T, A]")
    with tf.control_dependencies((rank_check,)):
        actions = tf.identity(actions)
    shape = tf.shape(actions)
    checks = (
        tf.debugging.assert_positive(shape[0], message="actions must be nonempty"),
        tf.debugging.assert_positive(shape[1], message="actions must be nonempty"),
        tf.debugging.assert_all_finite(actions, "actions must contain only finite values"),
    )
    with tf.control_dependencies(checks):
        return tf.identity(actions)


def _validated_tf_vector(value: object, *, name: str, length: Any, tf: Any) -> Any:
    vector = tf.convert_to_tensor(value)
    if not vector.dtype.is_floating:
        raise ValueError(f"{name} must use a real floating dtype")
    rank_check = tf.debugging.assert_rank(vector, 1, message=f"{name} must be rank 1")
    with tf.control_dependencies((rank_check,)):
        vector = tf.identity(vector)
    checks = (
        tf.debugging.assert_equal(tf.shape(vector)[0], length, message=f"{name} length must match actions"),
        tf.debugging.assert_all_finite(vector, f"{name} must contain only finite values"),
    )
    with tf.control_dependencies(checks):
        return tf.identity(vector)


def _tf_history(actions: Any, starts: Any, *, history_length: int, tf: Any) -> tuple[Any, Any]:
    offsets = tf.range(history_length, dtype=starts.dtype)[tf.newaxis, :]
    history_indices = starts[:, tf.newaxis] - history_length + offsets
    history_mask = history_indices >= 0
    values = tf.gather(actions, tf.maximum(history_indices, 0))
    values = tf.where(history_mask[..., tf.newaxis], values, tf.zeros_like(values))
    return values, history_mask


def _tf_gather_aligned_tree(
    tree: object,
    indices: Any,
    *,
    length: Any,
    name: str,
    tf: Any,
) -> Any:
    return tf.nest.map_structure(
        lambda value: _tf_gather_aligned_value(
            value,
            indices,
            length=length,
            name=name,
            tf=tf,
        ),
        tree,
    )


def _tf_gather_aligned_value(
    value: object,
    indices: Any,
    *,
    length: Any,
    name: str,
    tf: Any,
) -> Any:
    aligned = _tf_aligned_or_scalar(value, length=length, name=name, tf=tf)
    return tf.gather(aligned, indices)


def _tf_aligned_or_scalar(value: object, *, length: Any, name: str, tf: Any) -> Any:
    tensor = tf.convert_to_tensor(value)
    rank = tensor.shape.rank
    scalar_allowed = name in {"prompt", "episode_id", "frame_index"}
    if rank == 0:
        if not scalar_allowed:
            raise ValueError(f"{name} must have an episode axis")
        return tf.fill((length,), tensor)
    if rank is not None and rank != 1 and scalar_allowed:
        raise ValueError(f"{name} must be scalar or rank 1")
    if rank is None and scalar_allowed:
        rank_check = tf.debugging.assert_less_equal(tf.rank(tensor), 1, message=f"{name} must be scalar or rank 1")

        def broadcast_scalar() -> Any:
            return tf.fill((length,), tensor)

        def validate_aligned() -> Any:
            length_check = tf.debugging.assert_equal(
                tf.shape(tensor)[0],
                length,
                message=f"{name} length must match actions",
            )
            with tf.control_dependencies((length_check,)):
                return tf.identity(tensor)

        with tf.control_dependencies((rank_check,)):
            return tf.cond(tf.equal(tf.rank(tensor), 0), broadcast_scalar, validate_aligned)
    rank_check = tf.debugging.assert_rank_at_least(tensor, 1, message=f"{name} must have an episode axis")
    with tf.control_dependencies((rank_check,)):
        tensor = tf.identity(tensor)
    length_check = tf.debugging.assert_equal(
        tf.shape(tensor)[0],
        length,
        message=f"{name} length must match actions",
    )
    with tf.control_dependencies((length_check,)):
        return tf.identity(tensor)


def _tf_paired_record(
    traj: Mapping[str, Any],
    *,
    starts: Any,
    chunk_indices: Any,
    actions: Any,
    trajectory_length: Any,
    history_length: int,
    tf: Any,
) -> dict[str, Any]:
    history, history_mask = _tf_history(actions, starts, history_length=history_length, tf=tf)
    record = {
        "actions": tf.gather(actions, chunk_indices),
        "observation": _tf_gather_aligned_tree(
            traj["observation"],
            starts,
            length=trajectory_length,
            name="observation",
            tf=tf,
        ),
        "prompt": _tf_gather_aligned_value(
            traj["prompt"],
            starts,
            length=trajectory_length,
            name="prompt",
            tf=tf,
        ),
        "action_history": history,
        "action_history_mask": history_mask,
    }
    for key in ("rewards", "discounts", "executed_actions"):
        if key in traj:
            values = _tf_aligned_or_scalar(
                traj[key],
                length=trajectory_length,
                name=key,
                tf=tf,
            )
            record[key] = tf.gather(values, chunk_indices)
    for key in ("passes_filter", "adapter_mask"):
        if key in traj:
            record[key] = _tf_gather_aligned_value(
                traj[key],
                starts,
                length=trajectory_length,
                name=key,
                tf=tf,
            )
    if "episode_id" in traj:
        record["episode_id"] = _tf_gather_aligned_value(
            traj["episode_id"],
            starts,
            length=trajectory_length,
            name="episode_id",
            tf=tf,
        )
    frame_indices = traj.get("frame_index", tf.range(trajectory_length, dtype=tf.int32))
    record["frame_index"] = _tf_gather_aligned_value(
        frame_indices,
        starts,
        length=trajectory_length,
        name="frame_index",
        tf=tf,
    )
    return record


def _real_floating_array(value: object, *, name: str, rank: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != rank:
        raise ValueError(f"{name} must be rank {rank}")
    try:
        floating = np.issubdtype(array.dtype, np.floating)
    except TypeError:
        floating = False
    if not floating or not np.isrealobj(array):
        raise ValueError(f"{name} must use a real floating dtype")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array
