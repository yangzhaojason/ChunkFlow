"""RLDS-based DROID datasets with strict episode-local ChunkFlow streams."""

from enum import Enum
from enum import auto
import json
import logging
import numbers
from pathlib import Path
from typing import Literal

import tqdm

import openpi.shared.download as download
from openpi.training.chunkflow_rlds import build_tf_paired_chunks
from openpi.training.chunkflow_rlds import build_tf_step_transitions


class DroidActionSpace(Enum):
    """Action space for DROID dataset."""

    JOINT_POSITION = auto()
    JOINT_VELOCITY = auto()
    CARTESIAN_POSITION = auto()


def _validate_chunkflow_options(
    *,
    mode: object,
    action_chunk_size: object,
    stride: object,
    history_length: object,
    seed: object,
) -> tuple[Literal["legacy", "paired", "transition"], int, int | None, int, int]:
    if not isinstance(mode, str) or mode not in {"legacy", "paired", "transition"}:
        raise ValueError("chunkflow_mode must be one of: legacy, paired, transition")
    for name, value in (
        ("action_chunk_size", action_chunk_size),
        ("history_length", history_length),
        ("seed", seed),
    ):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise ValueError(f"{name} must be an integer")
    action_chunk_size = int(action_chunk_size)
    history_length = int(history_length)
    seed = int(seed)
    if action_chunk_size <= 0:
        raise ValueError("action_chunk_size must be positive")
    if not 0 <= history_length <= action_chunk_size:
        raise ValueError("history_length must satisfy 0 <= history_length <= action_chunk_size")
    if stride is not None:
        if isinstance(stride, bool) or not isinstance(stride, numbers.Integral):
            raise ValueError("chunkflow_stride must be an integer")
        stride = int(stride)
        if not 0 < stride <= action_chunk_size:
            raise ValueError("chunkflow_stride must satisfy 0 < stride <= action_chunk_size")
    if mode != "legacy" and stride is None:
        raise ValueError("chunkflow_stride is required in paired and transition modes")
    return mode, action_chunk_size, stride, history_length, seed


def _strict_reward_and_continuation(traj, *, tf):
    if "reward" not in traj:
        raise KeyError("transition mode requires the source reward tensor")
    rewards = tf.reshape(traj["reward"], [-1])
    if "discount" in traj:
        return rewards, tf.reshape(traj["discount"], [-1])

    terminal_signals = []
    if "is_terminal" in traj:
        terminal_signals.append(tf.reshape(traj["is_terminal"], [-1]))
    if "is_last" in traj:
        terminal_signals.append(tf.reshape(traj["is_last"], [-1]))
    if not terminal_signals:
        raise KeyError("transition mode requires source discount or explicit is_terminal/is_last")
    terminal = terminal_signals[0]
    for signal in terminal_signals[1:]:
        terminal = tf.logical_or(terminal, signal)
    return rewards, 1.0 - tf.cast(terminal, tf.float32)


def _build_filter_table(filter_dict_path, *, tf):
    if filter_dict_path is None:
        return tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer([""], [True]),
            default_value=True,
        )

    cached_filter_dict_path = download.maybe_download(filter_dict_path)
    with Path(cached_filter_dict_path).open("r") as file:
        filter_dict = json.load(file)
    logging.info("Using filter dictionary with %d episodes", len(filter_dict))

    keys = []
    values = []
    for episode_key, ranges in tqdm.tqdm(filter_dict.items(), desc="Creating idle filter hash table..."):
        for start, end in ranges:
            for frame in range(start, end):
                keys.append(f"{episode_key}--{frame}")
                values.append(True)
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, values),
        default_value=False,
    )
    logging.info("Filter hash table initialized")
    return table


def _build_droid_dataset(
    *,
    data_dir: str,
    batch_size: int,
    shuffle: bool,
    action_chunk_size: int,
    action_space: DroidActionSpace,
    shuffle_buffer_size: int,
    num_parallel_reads: int,
    num_parallel_calls: int,
    filter_dict_path,
    chunkflow_mode: Literal["legacy", "paired", "transition"],
    chunkflow_stride: int | None,
    history_length: int,
    seed: int,
    use_recorded_action: bool,
    observation_position_key: str,
):
    import dlimp as dl  # noqa: PLC0415
    import tensorflow as tf  # noqa: PLC0415
    import tensorflow_datasets as tfds  # noqa: PLC0415

    tf.config.set_visible_devices([], "GPU")
    builder = tfds.builder("droid_100", data_dir=data_dir, version="1.0.0")
    dataset = dl.DLataset.from_rlds(
        builder,
        split="train",
        shuffle=shuffle if chunkflow_mode == "legacy" else False,
        num_parallel_reads=num_parallel_reads,
    )
    dataset = dataset.filter(
        lambda traj: tf.strings.regex_full_match(
            traj["traj_metadata"]["episode_metadata"]["file_path"][0],
            ".*success.*",
        )
    )
    dataset = dataset.repeat()
    filter_table = _build_filter_table(filter_dict_path, tf=tf)

    def restructure(traj):
        if use_recorded_action:
            actions = traj["action"]
        else:
            arm_actions = (
                traj["action_dict"]["joint_position"]
                if action_space == DroidActionSpace.JOINT_POSITION
                else traj["action_dict"]["joint_velocity"]
            )
            actions = tf.concat((arm_actions, traj["action_dict"]["gripper_position"]), axis=-1)

        metadata = traj["traj_metadata"]["episode_metadata"]
        recording_path = metadata["recording_folderpath"][0]
        file_path = metadata["file_path"][0]
        stable_path = tf.strings.join((recording_path, "--", file_path))
        episode_id = tf.cast(
            tf.strings.to_hash_bucket_strong(
                stable_path,
                2_147_483_647,
                key=(0x12345678, 0x9ABCDEF0),
            ),
            tf.int32,
        )
        episode_seed = tf.random.experimental.stateless_fold_in(
            tf.constant((seed, 0), dtype=tf.int32),
            episode_id,
        )
        camera_seed = tf.random.experimental.stateless_fold_in(episode_seed, 0)
        language_seed = tf.random.experimental.stateless_fold_in(episode_seed, 1)
        exterior_img = tf.cond(
            tf.random.stateless_uniform((), seed=camera_seed) > 0.5,
            lambda: traj["observation"]["exterior_image_1_left"],
            lambda: traj["observation"]["exterior_image_2_left"],
        )
        instructions = tf.stack(
            (
                traj["language_instruction"],
                traj["language_instruction_2"],
                traj["language_instruction_3"],
            ),
            axis=0,
        )
        instruction = tf.random.experimental.stateless_shuffle(instructions, seed=language_seed)[0]

        trajectory_length = tf.shape(actions)[0]
        frame_index = tf.cast(traj["_frame_index"], tf.int32)
        string_indices = tf.as_string(tf.range(trajectory_length))
        step_id = (
            traj["traj_metadata"]["episode_metadata"]["recording_folderpath"]
            + "--"
            + traj["traj_metadata"]["episode_metadata"]["file_path"]
            + "--"
            + string_indices
        )
        passes_filter = filter_table.lookup(step_id)
        record = {
            "actions": actions,
            "observation": {
                "image": exterior_img,
                "wrist_image": traj["observation"]["wrist_image_left"],
                observation_position_key: traj["observation"][observation_position_key],
                "gripper_position": traj["observation"]["gripper_position"],
            },
            "prompt": instruction,
            "step_id": step_id,
            "passes_filter": passes_filter,
        }

        if chunkflow_mode == "legacy":
            rewards = (
                traj["reward"]
                if "reward" in traj
                else tf.zeros((trajectory_length,), dtype=tf.float32)
            )
            if "discount" in traj:
                discounts = traj["discount"]
            else:
                is_terminal = (
                    traj["is_terminal"]
                    if "is_terminal" in traj
                    else tf.zeros((trajectory_length,), dtype=tf.bool)
                )
                is_last = (
                    traj["is_last"]
                    if "is_last" in traj
                    else tf.zeros((trajectory_length,), dtype=tf.bool)
                )
                discounts = 1.0 - tf.cast(tf.logical_or(is_terminal, is_last), tf.float32)
            record["rewards"] = tf.reshape(rewards, [-1])
            record["discounts"] = tf.reshape(discounts, [-1])
            return record

        record.update(
            executed_actions=actions,
            episode_id=tf.fill((trajectory_length,), episode_id),
            frame_index=frame_index,
        )
        if chunkflow_mode == "transition":
            rewards, discounts = _strict_reward_and_continuation(traj, tf=tf)
            record["rewards"] = rewards
            record["discounts"] = discounts
        else:
            if "reward" in traj:
                record["rewards"] = tf.reshape(traj["reward"], [-1])
            if "discount" in traj:
                record["discounts"] = tf.reshape(traj["discount"], [-1])
        return record

    dataset = dataset.traj_map(restructure, num_parallel_calls)

    def chunk_actions(traj):
        trajectory_length = tf.shape(traj["actions"])[0]
        chunk_indices = tf.range(action_chunk_size)[None, :] + tf.range(trajectory_length)[:, None]
        chunk_indices = tf.minimum(chunk_indices, trajectory_length - 1)
        traj["actions"] = tf.gather(traj["actions"], chunk_indices)
        traj["rewards"] = tf.gather(traj["rewards"], chunk_indices)
        traj["discounts"] = tf.gather(traj["discounts"], chunk_indices)
        return traj

    def filter_from_dict(frame):
        return frame["passes_filter"]

    def remove_passes_filter(frame):
        frame.pop("passes_filter")
        return frame

    def filter_paired_from_dict(pair):
        return pair["current"]["passes_filter"]

    def remove_paired_passes_filter(pair):
        pair["previous"].pop("passes_filter")
        pair["current"].pop("passes_filter")
        return pair

    def filter_transition_from_dict(transition):
        return transition["current"]["passes_filter"]

    def remove_transition_passes_filter(transition):
        transition["current"].pop("passes_filter")
        transition["next"].pop("passes_filter")
        return transition

    def decode_record(record):
        record["observation"]["image"] = tf.io.decode_image(
            record["observation"]["image"],
            expand_animations=False,
            dtype=tf.uint8,
        )
        record["observation"]["wrist_image"] = tf.io.decode_image(
            record["observation"]["wrist_image"],
            expand_animations=False,
            dtype=tf.uint8,
        )
        return record

    if chunkflow_mode == "legacy":
        dataset = dataset.traj_map(chunk_actions, num_parallel_calls)
        dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)
        dataset = dataset.filter(filter_from_dict)
        dataset = dataset.map(remove_passes_filter, num_parallel_calls=num_parallel_calls)
        dataset = dataset.frame_map(decode_record, num_parallel_calls)
    elif chunkflow_mode == "paired":
        dataset = dataset.traj_map(
            lambda traj: build_tf_paired_chunks(
                traj,
                horizon=action_chunk_size,
                stride=chunkflow_stride,
                history_length=history_length,
                tf=tf,
            ),
            num_parallel_calls,
        )
        dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)
        dataset = dataset.filter(filter_paired_from_dict)
        dataset = dataset.map(remove_paired_passes_filter, num_parallel_calls=num_parallel_calls)
        dataset = dataset.frame_map(
            lambda pair: {
                "previous": decode_record(pair["previous"]),
                "current": decode_record(pair["current"]),
            },
            num_parallel_calls,
        )
    else:
        dataset = dataset.traj_map(
            lambda traj: build_tf_step_transitions(
                traj,
                history_length=history_length,
                tf=tf,
            ),
            num_parallel_calls,
        )
        dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)
        dataset = dataset.filter(filter_transition_from_dict)
        dataset = dataset.map(remove_transition_passes_filter, num_parallel_calls=num_parallel_calls)

        def decode_transition(transition):
            transition["current"] = decode_record(transition["current"])
            transition["next"] = decode_record(transition["next"])
            return transition

        dataset = dataset.frame_map(decode_transition, num_parallel_calls)

    dataset = dataset.shuffle(shuffle_buffer_size, seed=seed)
    dataset = dataset.batch(batch_size)
    return dataset.with_ram_budget(1), filter_table


class DroidRldsDataset:
    supports_chunkflow_pairs = True
    supports_chunkflow_transitions = True

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        *,
        shuffle: bool = True,
        action_chunk_size: int = 16,
        action_space: DroidActionSpace = DroidActionSpace.JOINT_POSITION,
        max_loaded_steps_per_episode: int = 100,
        shuffle_buffer_size: int = 250_000,
        num_parallel_reads: int = -1,
        num_parallel_calls: int = -1,
        filter_dict_path=None,
        chunkflow_mode: Literal["legacy", "paired", "transition"] = "legacy",
        chunkflow_stride: int | None = None,
        history_length: int = 0,
        seed: int = 0,
    ):
        del max_loaded_steps_per_episode
        chunkflow_mode, action_chunk_size, chunkflow_stride, history_length, seed = _validate_chunkflow_options(
            mode=chunkflow_mode,
            action_chunk_size=action_chunk_size,
            stride=chunkflow_stride,
            history_length=history_length,
            seed=seed,
        )
        self.dataset, self.filter_table = _build_droid_dataset(
            data_dir=data_dir,
            batch_size=batch_size,
            shuffle=shuffle,
            action_chunk_size=action_chunk_size,
            action_space=action_space,
            shuffle_buffer_size=shuffle_buffer_size,
            num_parallel_reads=num_parallel_reads,
            num_parallel_calls=num_parallel_calls,
            filter_dict_path=filter_dict_path,
            chunkflow_mode=chunkflow_mode,
            chunkflow_stride=chunkflow_stride,
            history_length=history_length,
            seed=seed,
            use_recorded_action=False,
            observation_position_key="joint_position",
        )
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        yield from self.dataset.as_numpy_iterator()

    def __len__(self):
        return 20_000_000


class DroidRldsNewDataset:
    supports_chunkflow_pairs = True
    supports_chunkflow_transitions = True

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        *,
        shuffle: bool = True,
        action_chunk_size: int = 16,
        action_space: DroidActionSpace = DroidActionSpace.CARTESIAN_POSITION,
        max_loaded_steps_per_episode: int = 100,
        shuffle_buffer_size: int = 250_000,
        num_parallel_reads: int = -1,
        num_parallel_calls: int = -1,
        filter_dict_path=None,
        chunkflow_mode: Literal["legacy", "paired", "transition"] = "legacy",
        chunkflow_stride: int | None = None,
        history_length: int = 0,
        seed: int = 0,
    ):
        del max_loaded_steps_per_episode
        chunkflow_mode, action_chunk_size, chunkflow_stride, history_length, seed = _validate_chunkflow_options(
            mode=chunkflow_mode,
            action_chunk_size=action_chunk_size,
            stride=chunkflow_stride,
            history_length=history_length,
            seed=seed,
        )
        self.dataset, self.filter_table = _build_droid_dataset(
            data_dir=data_dir,
            batch_size=batch_size,
            shuffle=shuffle,
            action_chunk_size=action_chunk_size,
            action_space=action_space,
            shuffle_buffer_size=shuffle_buffer_size,
            num_parallel_reads=num_parallel_reads,
            num_parallel_calls=num_parallel_calls,
            filter_dict_path=filter_dict_path,
            chunkflow_mode=chunkflow_mode,
            chunkflow_stride=chunkflow_stride,
            history_length=history_length,
            seed=seed,
            use_recorded_action=True,
            observation_position_key="cartesian_position",
        )
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        yield from self.dataset.as_numpy_iterator()

    def __len__(self):
        return 20_000_000
