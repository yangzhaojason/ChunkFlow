"""
RLDS-based data loader for DROID.
While openpi typically uses LeRobot's data loader, it is not currently scalable enough for larger datasets like DROID.
Thus, we provide a data loader example here that uses the RLDS data format.
The data loader also applies a few DROID-specific data filters / transformations.
"""

from enum import Enum
from enum import auto
import json
import logging
import os
from pathlib import Path
import pickle

import numpy as np
import tqdm

import openpi.shared.download as download


def paired_episode_action_windows(
    actions: np.ndarray,
    *,
    horizon: int,
    stride: int,
    history_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reference NumPy assembly for full, episode-local ChunkFlow windows."""

    actions = np.asarray(actions)
    if actions.ndim != 2:
        raise ValueError("actions must be rank 2 [trajectory_length, action_dim]")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if stride <= 0 or stride > horizon:
        raise ValueError("stride must satisfy 0 < stride <= horizon")
    if not 0 <= history_length <= horizon:
        raise ValueError("history_length must satisfy 0 <= history_length <= horizon")

    pair_count = max(actions.shape[0] - stride - horizon + 1, 0)
    starts = np.arange(0, pair_count, stride, dtype=np.int64)
    offsets = np.arange(horizon, dtype=np.int64)
    previous = actions[starts[:, None] + offsets[None, :]]
    current = actions[starts[:, None] + stride + offsets[None, :]]

    def gather_history(starts_to_condition: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        indices = starts_to_condition[:, None] - history_length + np.arange(history_length)[None, :]
        mask = indices >= 0
        safe = np.maximum(indices, 0)
        values = actions[safe]
        values = np.where(mask[..., None], values, np.zeros((), dtype=actions.dtype))
        return values, mask

    previous_history, previous_mask = gather_history(starts)
    current_history, current_mask = gather_history(starts + stride)
    return starts, previous, current, previous_history, previous_mask, current_history, current_mask


class TruthActionSpace(Enum):
    """Action space for Truth dataset."""

    JOINT_POSITION = auto()
    JOINT_VELOCITY = auto()
    CARTESIAN_POSITION = auto()


class TruthRldsDataset:
    def __init__(
        self,
        repo_id: str,
        data_dir: str,
        batch_size: int,
        *,  # Force keyword-only arguments
        shuffle: bool = True,
        action_chunk_size: int = 16,
        # We default to joint position actions, since they allow policy evaluation in simulation.
        action_space: TruthActionSpace = TruthActionSpace.JOINT_POSITION,
        max_loaded_steps_per_episode: int = 100,
        # Reduce this if you are running out of memory, but careful -- below ~100k shuffling is not sufficiently random.
        shuffle_buffer_size: int = 10_000,
        # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        num_parallel_reads: int = -1,
        # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        num_parallel_calls: int = -1,
        filter_dict_path=None,  # Path to json file with indices to sample during training
    ):
        # Import tensorflow here to not make it mandatory in case RLDS data loader is not used.
        import os

        import dlimp as dl
        import tensorflow as tf
        import tensorflow_datasets as tfds

        # Configure Tensorflow with *no GPU devices* (to prevent clobber with PyTorch / JAX)
        tf.config.set_visible_devices([], "GPU")

        logging.info("TFDS builder starting...")
        builder = tfds.builder(repo_id, data_dir=data_dir)
        dataset = dl.DLataset.from_rlds(
            builder,
            split="train",
            shuffle=shuffle,
            num_parallel_reads=num_parallel_reads,
        )

        # Filter out any unsuccessful trajectories -- we use the file name to check this
        # todo 不过滤掉不成功的轨迹  因为已经筛选成功的轨迹了
        # dataset = dataset.filter(     q
        #     lambda traj: tf.strings.regex_full_match(
        #         traj["traj_metadata"]["episode_metadata"]["file_path"][0], ".*success.*"
        #     )
        # )

        # Repeat 将在过滤/子集选择完成后再调用，避免空集导致下游长时间等待

        # Load the filter dictionary if provided.
        # The filter dictionary is a JSON file that maps episode keys to ranges of frames to sample
        # (e.g.,
        # {
        #     "<episode key>": [[0, 100], [200, 300]]
        # }
        # means keep frames 0-99 and 200-299).
        if filter_dict_path is not None:
            cached_filter_dict_path = download.maybe_download(filter_dict_path)
            with Path(cached_filter_dict_path).open("r") as f:
                filter_dict = json.load(f)

            logging.info(
                f"Using filter dictionary with {len(filter_dict)} episodes")

            keys_tensor = []
            values_tensor = []

            for episode_key, ranges in tqdm.tqdm(filter_dict.items(), desc="Creating idle filter hash table..."):
                for start, end in ranges:
                    for t in range(start, end):
                        frame_key = f"{episode_key}--{t}"
                        keys_tensor.append(frame_key)
                        values_tensor.append(True)
            self.filter_table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor), default_value=False
            )
            logging.info("Filter hash table initialized")
        else:
            self.filter_table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer([""], [True]), default_value=True
            )

        def restructure(traj):
            """Reformat observation and action keys, sample language instruction."""
            # Important: we use joint *position* action space -- easier to simulate!
            actions = tf.concat(
                (
                    (
                        traj["action_dict"]["joint_position"]
                        if action_space == TruthActionSpace.JOINT_POSITION
                        else traj["action_dict"]["joint_velocity"]
                    ),
                    traj["action_dict"]["gripper_position"],
                ),
                axis=-1,
            )
            # Randomly samples one of the two exterior images in DROID during training (we only train with one at a time).
            # Note: the "left" refers to the left camera in the stereo pair, we only train on the left camera.
            # exterior_img = tf.cond(
            #     tf.random.uniform(shape=[]) > 0.5,
            #     lambda: traj["observation"]["exterior_image_1_left"],
            #     lambda: traj["observation"]["exterior_image_2_left"],
            # )
            exterior_img = traj["observation"]["exterior_image_1_left"]
            wrist_img = traj["observation"]["wrist_image_left"]
            # Randomly sample one of the three language instructions
            instruction = tf.random.shuffle(
                [
                    traj["language_instruction"],
                    # traj["language_instruction_2"],
                    # traj["language_instruction_3"],
                ]
            )[0]

            # --- Rewards & discounts (for AWAC). If not present, create safe defaults. ---
            traj_len = tf.shape(actions)[0]
            # Default sparse terminal reward/discount if missing: last step = 1.0 reward, 0.0 discount
            rewards = traj["reward"] if "reward" in traj else tf.concat(
                [tf.zeros((tf.maximum(traj_len - 1, 0),), tf.float32), tf.ones((1,), tf.float32)], axis=0
            )
            if "discount" in traj:
                discounts = traj["discount"]
            else:
                discounts = tf.concat(
                    [tf.ones((tf.maximum(traj_len - 1, 0),), tf.float32), tf.zeros((1,), tf.float32)], axis=0
                )

            # traj_len = tf.shape(traj["action"])[0]
            # indices = tf.as_string(tf.range(traj_len))

            # Data filtering:
            # Compute a uniquely-identifying step ID by concatenating the recording folderpath, file path,
            # and each step's time step index. This will index into the filter hash table, and if it returns true,
            # then the frame passes the filter.
            # step_id = (
            #     traj["traj_metadata"]["episode_metadata"]["recording_folderpath"]
            #     + "--"
            #     + traj["traj_metadata"]["episode_metadata"]["file_path"]
            #     + "--"
            #     + indices
            # )
            # passes_filter = self.filter_table.lookup(step_id)

            return {
                "actions": actions,
                "observation": {
                    "image": exterior_img,
                    "wrist_image": wrist_img,
                    "joint_position": traj["observation"]["joint_position"],
                    "gripper_position": traj["observation"]["gripper_position"],
                },
                "prompt": instruction,
                "rewards": tf.reshape(rewards, [-1]),
                "discounts": tf.reshape(discounts, [-1]),
                # "step_id": step_id,
                # "passes_filter": passes_filter,
            }

        dataset = dataset.traj_map(restructure, num_parallel_calls)

        def chunk_actions(traj):
            """Splits episode into action chunks."""
            traj_len = tf.shape(traj["actions"])[0]

            # For each step in the trajectory, construct indices for the next n actions
            action_chunk_indices = tf.broadcast_to(
                tf.range(action_chunk_size)[None],
                [traj_len, action_chunk_size],
            ) + tf.broadcast_to(
                tf.range(traj_len)[:, None],
                [traj_len, action_chunk_size],
            )

            # Cap to length of the sequence --> final chunks will repeat the last action
            # This makes sense, since we are using absolute joint + gripper position actions
            action_chunk_indices = tf.minimum(
                action_chunk_indices, traj_len - 1)

            # Gather the actions for each chunk
            traj["actions"] = tf.gather(traj["actions"], action_chunk_indices)
            # Align rewards/discounts with the same chunk indices
            if "rewards" in traj:
                traj["rewards"] = tf.gather(traj["rewards"], action_chunk_indices)
            if "discounts" in traj:
                traj["discounts"] = tf.gather(traj["discounts"], action_chunk_indices)
            return traj

        dataset = dataset.traj_map(chunk_actions, num_parallel_calls)

        def filter_idle(traj):
            """Filter out chunks with idle actions.
            --> we filter if at least first half of chunk does not move.
            """
            if action_space == TruthActionSpace.JOINT_POSITION:
                # Compute delta to first position in action chunk
                return tf.reduce_any(
                    tf.abs(
                        traj["actions"][: action_chunk_size // 2] -
                        traj["actions"][:1]
                    )
                    > 1e-3
                )
            return tf.reduce_any(
                tf.abs(traj["actions"][: action_chunk_size // 2]) > 1e-3
            )

        dataset = dataset.filter(filter_idle)

        # Flatten: map from trajectory dataset to dataset of individual action chunks
        dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)

        # Decode images: RLDS saves encoded images, only decode now for efficiency
        def decode_images(traj):
            traj["observation"]["image"] = tf.io.decode_image(
                traj["observation"]["image"], expand_animations=False, dtype=tf.uint8
            )
            traj["observation"]["wrist_image"] = tf.io.decode_image(
                traj["observation"]["wrist_image"],
                expand_animations=False,
                dtype=tf.uint8,
            )
            return traj

        dataset = dataset.frame_map(decode_images, num_parallel_calls)

        # Shuffle, batch
        dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.batch(batch_size)
        # Note =>> Seems to reduce memory usage without affecting speed?
        dataset = dataset.with_ram_budget(1)

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        yield from self.dataset.as_numpy_iterator()

    def __len__(self):
        # This is the approximate number of samples in DROID after filtering.
        # Easier to hardcode than to iterate through the dataset and compute it.
        return 20_000_000


def load_all_frame_idxs_pickle(file_path):
    """从 Pickle 文件加载 all_frame_idxs"""
    with open(file_path, 'rb') as f:
        all_frame_idxs = pickle.load(f)

    return all_frame_idxs


def load_segment_index(
    data_dir: str,
    repo_id: str,
    *,
    path_exists=os.path.isfile,
    loader=load_all_frame_idxs_pickle,
):
    """Load the repository-relative segment index used for repeated sampling."""
    segment_index_path = os.path.join(
        data_dir,
        f"{repo_id}_all_frame_idxs_add_path.pkl",
    )
    if not path_exists(segment_index_path):
        raise FileNotFoundError(
            f"Segment index not found: {segment_index_path}. "
            "Generate it first or disable downsampled_and_repeated."
        )
    return loader(segment_index_path)


class TruthRldsDatasetCartesian:
    def __init__(
        self,
        repo_id: str,
        data_dir: str,
        batch_size: int,
        *,  # Force keyword-only arguments
        shuffle: bool = True,
        action_chunk_size: int = 16,
        # We default to joint position actions, since they allow policy evaluation in simulation.
        action_space: TruthActionSpace = TruthActionSpace.CARTESIAN_POSITION,
        max_loaded_steps_per_episode: int = 100,
        # Reduce this if you are running out of memory, but careful -- below ~100k shuffling is not sufficiently random.
        shuffle_buffer_size: int = 10_000,  # todo 原始为250000
        # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        num_parallel_reads: int = -1,
        # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        num_parallel_calls: int = -1,
        filter_dict_path=None,
        downsampled_and_repeated: bool = False,
        segments_debug_samples: int = 0,
    ):
        # 预定义片段预计算变量，避免作用域未定义
        precomputed_segment_keys = None
        precomputed_segment_values = None
        precomputed_segment_keys_count = None

        # Import tensorflow here to not make it mandatory in case RLDS data loader is not used.
        import dlimp as dl
        import numpy as np
        import tensorflow as tf
        import tensorflow_datasets as tfds

        # Configure Tensorflow with *no GPU devices* (to prevent clobber with PyTorch / JAX)
        tf.config.set_visible_devices([], "GPU")

        logging.info("TFDS builder starting (cartesian)...")
        logging.info("repo_id: %s, data_dir: %s", repo_id, data_dir)
        builder = tfds.builder(repo_id, data_dir=data_dir)
        dataset = dl.DLataset.from_rlds(
            builder,
            split="train",
            shuffle=shuffle,
            num_parallel_reads=num_parallel_reads,
        )

        # Repeat dataset so we never run out of data.
        dataset = dataset.repeat()

        # 当启用片段级重复采样时，加载 all_frame_idxs，并构造 step 级别匹配表
        result_ds = None
        logging.info("downsampled_and_repeated: %s", downsampled_and_repeated)
        if downsampled_and_repeated:
            # 默认从 data_dir/repo_id_all_frame_idxs_add_path.pkl 加载
            if precomputed_segment_keys is not None:
                keys_tensor = precomputed_segment_keys
                values_tensor = precomputed_segment_values
                segment_keys_count = precomputed_segment_keys_count
                logging.info(f"reuse precomputed keys: {segment_keys_count}")
            else:
                all_frame_idxs = load_segment_index(data_dir, repo_id)

            # 适配 DLataset：每个元素为一个 episode，字段时间维度为第0维（形状 (None, ...））。
            # 做法：按 file_path 构建区间映射；基于 file_path 与 _frame_index 生成掩码；
            #       对所有时间序列字段在 axis=0 上应用 boolean_mask；过滤空样本。

            """数据下采样（仅保留 frame_idxs 数量不超过 1 的 episode）"""
            def _to_py_str(v):
                try:
                    if isinstance(v, tf.Tensor):
                        v = v.numpy()
                except Exception:
                    pass
                if isinstance(v, (bytes, bytearray)):
                    return v.decode("utf-8")
                return str(v)

            # 1) 构建 file_path -> intervals 映射，并计算每个 file_path 的区间数量
            intervals_by_path = {}
            num_intervals_by_path = {}
            for rec in all_frame_idxs:
                fp = _to_py_str(rec.get("file_path", ""))
                intervals = rec["frame_idxs"] or []
                intervals_np = np.array(
                    intervals, dtype=np.int64).reshape(-1, 2)
                intervals_by_path[fp] = intervals_np
                num_intervals_by_path[fp] = intervals_np.shape[0]

            # 2) 仅保留区间数 <= 2 的 file_path
            allowed_fps = {fp for fp,
                           n in num_intervals_by_path.items() if n < 2}

            # 3) 掩码构造函数（仅使用允许的 file_path；区间内为 True）
            def build_mask_by_fp_py(file_path_scalar, frame_index_eager):
                if hasattr(file_path_scalar, "numpy"):
                    fp_val = file_path_scalar.numpy()
                else:
                    fp_val = file_path_scalar
                if isinstance(fp_val, (bytes, bytearray)):
                    fp = fp_val.decode("utf-8")
                else:
                    fp = str(fp_val)

                if hasattr(frame_index_eager, "numpy"):
                    frame_idx = frame_index_eager.numpy().astype(np.int64)
                else:
                    frame_idx = np.asarray(frame_index_eager, dtype=np.int64)

                if frame_idx.shape[0] == 0 or fp not in allowed_fps:
                    return np.zeros((frame_idx.shape[0],), dtype=np.bool_)

                mask = np.zeros(frame_idx.shape[0], dtype=np.bool_)
                intervals = intervals_by_path.get(
                    fp, np.empty((0, 2), dtype=np.int64))
                for s, e in intervals:
                    if e > s:
                        mask |= (frame_idx >= s) & (frame_idx < e)
                return mask

            def apply_mask_structure(example):
                frame_index_vec = example["_frame_index"]
                file_path_vec = example["traj_metadata"]["episode_metadata"]["file_path"]
                file_path0 = tf.gather(file_path_vec, 0)

                mask = tf.py_function(build_mask_by_fp_py, [
                                      file_path0, frame_index_vec], Tout=tf.bool)
                mask.set_shape([None])

                def mask_tensor(t: tf.Tensor) -> tf.Tensor:
                    rank = t.shape.rank
                    if rank is None or rank == 0:
                        return t
                    return tf.boolean_mask(t, mask, axis=0)

                return tf.nest.map_structure(mask_tensor, example)

            masked_ds = dataset.map(
                apply_mask_structure, num_parallel_calls=tf.data.AUTOTUNE)

            # 4) 过滤空 episode
            def has_any_step(example):
                return tf.greater(tf.shape(example["_frame_index"])[0], 0)

            result_ds = masked_ds.filter(has_any_step)

            # todo 改为使用downsample data
            dataset = result_ds

        # Load the filter dictionary if provided.
        # The filter dictionary is a JSON file that maps episode keys to ranges of frames to sample
        # (e.g.,
        # {
        #     "<episode key>": [[0, 100], [200, 300]]
        # }
        # means keep frames 0-99 and 200-299).
        if filter_dict_path is not None:
            cached_filter_dict_path = download.maybe_download(filter_dict_path)
            with Path(cached_filter_dict_path).open("r") as f:
                filter_dict = json.load(f)

            logging.info(
                f"Using filter dictionary with {len(filter_dict)} episodes")

            keys_tensor = []
            values_tensor = []

            for episode_key, ranges in tqdm.tqdm(filter_dict.items(), desc="Creating idle filter hash table..."):
                for start, end in ranges:
                    for t in range(start, end):
                        frame_key = f"{episode_key}--{t}"
                        keys_tensor.append(frame_key)
                        values_tensor.append(True)
            self.filter_table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor), default_value=False
            )
            logging.info("Filter hash table initialized")
        else:
            self.filter_table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer([""], [True]), default_value=True
            )

        def restructure(traj):
            """Reformat observation and action keys, sample language instruction."""
            # Important: we use joint *position* action space -- easier to simulate!
            actions = tf.concat(
                (
                    (
                        traj["action_dict"]["cartesian_position"]
                        if action_space == TruthActionSpace.CARTESIAN_POSITION
                        else traj["action_dict"]["cartesian_velocity"]
                    ),
                    traj["action_dict"]["gripper_position"],
                ),
                axis=-1,
            )
            # Randomly samples one of the two exterior images in DROID during training (we only train with one at a time).
            # Note: the "left" refers to the left camera in the stereo pair, we only train on the left camera.
            # todo 由于现在采用的是三视角的图片，所以需要随机选择一个视角的图片
            exterior_img = tf.cond(
                tf.random.uniform(shape=[]) > 0.5,
                lambda: traj["observation"]["exterior_image_1_left"],
                lambda: traj["observation"]["exterior_image_top"],
            )
            # exterior_img = traj["observation"]["exterior_image_1_left"]
            wrist_img = traj["observation"]["wrist_image_left"]
            # Randomly sample one of the three language instructions
            # todo 这里需要自己设定language_instruction
            # instruction = tf.random.shuffle(
            #     [
            #         traj["language_instruction"],
            #         # traj["language_instruction_2"],
            #         # traj["language_instruction_3"],
            #     ]
            # )[0]
            # 注意：后续会调用 flatten -> from_tensor_slices(traj)，要求每个字段在时间维度拥有 rank>=1。
            # 因此需要将文本提示扩展为按时间步复制的 1D 向量，与其他序列字段长度一致。
            traj_len = tf.shape(traj["observation"]["cartesian_position"])[0]
            # instruction = tf.fill(
            #     [traj_len],
            #     "Grasp the wooden block and place it into the square recess.",
            # )
            instruction = tf.fill(
                [traj_len],
                "Pick up a single cloth strip and place it in the appropriate position.",
            )

            # --- Rewards & discounts (for AWAC). Prefer dataset rewards; if they are all zeros/ones, fall back to sparse terminal signal. ---
            traj_len = tf.shape(traj["observation"]["cartesian_position"])[0]

            # Default sparse terminal reward/discount: last step = 1.0 reward, 0.0 discount
            default_rewards = tf.concat(
                [tf.zeros((tf.maximum(traj_len - 1, 0),), tf.float32), tf.ones((1,), tf.float32)],
                axis=0,
            )
            default_discounts = tf.concat(
                [tf.ones((tf.maximum(traj_len - 1, 0),), tf.float32), tf.zeros((1,), tf.float32)],
                axis=0,
            )

            if "reward" in traj:
                rewards_raw = tf.reshape(traj["reward"], [-1])
                all_zero = tf.reduce_all(tf.equal(rewards_raw, 0.0))
                # 如果数据集中reward全为0，则退化为默认终止奖励
                rewards = tf.cond(all_zero, lambda: default_rewards, lambda: rewards_raw)
            else:
                rewards = default_rewards

            if "discount" in traj:
                discounts_raw = tf.reshape(traj["discount"], [-1])
                all_one = tf.reduce_all(tf.equal(discounts_raw, 1.0))
                # 如果数据集中discount全为1，则退化为默认终止折扣
                discounts = tf.cond(all_one, lambda: default_discounts, lambda: discounts_raw)
            else:
                discounts = default_discounts

            return {
                "actions": actions,
                "observation": {
                    "image": exterior_img,
                    "wrist_image": wrist_img,
                    "cartesian_position": traj["observation"]["cartesian_position"],
                    "gripper_position": traj["observation"]["gripper_position"],
                },
                "prompt": instruction,
                "rewards": tf.reshape(rewards, [-1]),
                "discounts": tf.reshape(discounts, [-1]),
            }

        dataset = dataset.traj_map(restructure, num_parallel_calls)

        def chunk_actions(traj):
            """Splits episode into action chunks."""
            traj_len = tf.shape(traj["actions"])[0]

            # For each step in the trajectory, construct indices for the next n actions
            action_chunk_indices = tf.broadcast_to(
                tf.range(action_chunk_size)[None],
                [traj_len, action_chunk_size],
            ) + tf.broadcast_to(
                tf.range(traj_len)[:, None],
                [traj_len, action_chunk_size],
            )

            # Cap to length of the sequence --> final chunks will repeat the last action
            # This makes sense, since we are using absolute joint + gripper position actions
            action_chunk_indices = tf.minimum(
                action_chunk_indices, traj_len - 1)

            # Gather the actions for each chunk
            traj["actions"] = tf.gather(traj["actions"], action_chunk_indices)
            if "rewards" in traj:
                traj["rewards"] = tf.gather(traj["rewards"], action_chunk_indices)
            if "discounts" in traj:
                traj["discounts"] = tf.gather(traj["discounts"], action_chunk_indices)
            return traj

        dataset = dataset.traj_map(chunk_actions, num_parallel_calls)

        def filter_idle(traj):
            """Filter out chunks with idle actions.
            --> we filter if at least first half of chunk does not move.
            """
            if action_space == TruthActionSpace.CARTESIAN_POSITION:
                # Compute delta to first position in action chunk
                return tf.reduce_any(
                    tf.abs(
                        traj["actions"][: action_chunk_size // 2] -
                        traj["actions"][:1]
                    )
                    > 1e-3
                )
            return tf.reduce_any(
                tf.abs(traj["actions"][: action_chunk_size // 2]) > 1e-3
            )

        dataset = dataset.filter(filter_idle)

        # Flatten: map from trajectory dataset to dataset of individual action chunks
        dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)

        # Decode images: RLDS saves encoded images, only decode now for efficiency
        def decode_images(traj):
            traj["observation"]["image"] = tf.io.decode_image(
                traj["observation"]["image"], expand_animations=False, dtype=tf.uint8
            )
            traj["observation"]["wrist_image"] = tf.io.decode_image(
                traj["observation"]["wrist_image"],
                expand_animations=False,
                dtype=tf.uint8,
            )
            return traj

        dataset = dataset.frame_map(decode_images, num_parallel_calls)

        # Shuffle, batch
        dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.batch(batch_size)
        # Note =>> Seems to reduce memory usage without affecting speed?
        dataset = dataset.with_ram_budget(1)

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        yield from self.dataset.as_numpy_iterator()

    def __len__(self):
        # This is the approximate number of samples in DROID after filtering.
        # Easier to hardcode than to iterate through the dataset and compute it.
        return 20_000_000



class TruthRldsDatasetJointWithoutGripper:
    supports_chunkflow_pairs = True

    def __init__(
        self,
        repo_id: str,
        data_dir: str,
        batch_size: int,
        *,  # Force keyword-only arguments
        shuffle: bool = True,
        action_chunk_size: int = 16,
        # We default to joint position actions, since they allow policy evaluation in simulation.
        action_space: TruthActionSpace = TruthActionSpace.JOINT_POSITION,
        max_loaded_steps_per_episode: int = 100,
        # Reduce this if you are running out of memory, but careful -- below ~100k shuffling is not sufficiently random.
        shuffle_buffer_size: int = 10_000,
        # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        num_parallel_reads: int = -1,
        # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        num_parallel_calls: int = -1,
        filter_dict_path=None,  # Path to json file with indices to sample during training
        chunkflow_stride: int | None = None,
        history_length: int = 0,
    ):
        # Import tensorflow here to not make it mandatory in case RLDS data loader is not used.
        import dlimp as dl
        import tensorflow as tf
        import tensorflow_datasets as tfds

        # Configure Tensorflow with *no GPU devices* (to prevent clobber with PyTorch / JAX)
        tf.config.set_visible_devices([], "GPU")

        logging.info("TFDS builder starting...")
        builder = tfds.builder(repo_id, data_dir=data_dir)
        dataset = dl.DLataset.from_rlds(
            builder,
            split="train",
            shuffle=shuffle,
            num_parallel_reads=num_parallel_reads,
        )

        if filter_dict_path is not None:
            cached_filter_dict_path = download.maybe_download(filter_dict_path)
            with Path(cached_filter_dict_path).open("r") as f:
                filter_dict = json.load(f)

            logging.info(
                f"Using filter dictionary with {len(filter_dict)} episodes")

            keys_tensor = []
            values_tensor = []

            for episode_key, ranges in tqdm.tqdm(filter_dict.items(), desc="Creating idle filter hash table..."):
                for start, end in ranges:
                    for t in range(start, end):
                        frame_key = f"{episode_key}--{t}"
                        keys_tensor.append(frame_key)
                        values_tensor.append(True)
            self.filter_table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor), default_value=False
            )
            logging.info("Filter hash table initialized")
        else:
            self.filter_table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer([""], [True]), default_value=True
            )

        def restructure(traj):
            """Reformat observation and action keys, sample language instruction."""

            actions = traj["action_dict"]["r_joint_position"]

            exterior_img = traj["observation"]["front_image"]
            wrist_img = traj["observation"]["top_image"]



            # 因此需要将文本提示扩展为按时间步复制的 1D 向量，与其他序列字段长度一致。
            traj_len = tf.shape(traj["observation"]["r_joint_position"])[0]


            # IMPORTANT: this function runs inside tf.data graph; do NOT use Python "in" with tensors.
            # `language_instruction` is typically a scalar tf.string tensor.
            #
            # NOTE: Some TF builds do not expose `tf.strings.find`; use regex-based "contains" instead.
            lang = tf.convert_to_tensor(traj["language_instruction"], dtype=tf.string)
            # Ensure 1D for matching; `language_instruction` may be scalar or vector depending on dataset.
            lang_vec = tf.reshape(lang, [-1])
            has_007 = tf.reduce_any(tf.strings.regex_full_match(lang_vec, ".*007.*"))

            def _mapped_instruction() -> tf.Tensor:
                return tf.fill(
                    [traj_len],
                    "Move the arm to the appropriate position.",
                )

            def _fallback_instruction() -> tf.Tensor:
                # Fallback: keep the original instruction (broadcast to trajectory length) so pipeline doesn't crash.
                # If you prefer hard failure, replace this with a tf.debugging.Assert.
                return tf.fill([traj_len], lang_vec[0])

            instruction = tf.cond(has_007, _mapped_instruction, _fallback_instruction)

            return {
                "actions": actions,
                "observation": {
                    "image": exterior_img,
                    "wrist_image": wrist_img,
                    "joint_position": traj["observation"]["r_joint_position"],
                },
                "prompt": instruction,
            }

        dataset = dataset.traj_map(restructure, num_parallel_calls)

        if chunkflow_stride is not None:
            if not 0 < chunkflow_stride <= action_chunk_size:
                raise ValueError("chunkflow_stride must satisfy 0 < stride <= action_chunk_size")
            if not 0 <= history_length <= action_chunk_size:
                raise ValueError("history_length must satisfy 0 <= history_length <= action_chunk_size")

        def chunk_actions(traj):
            """Splits episode into action chunks."""
            traj_len = tf.shape(traj["actions"])[0]

            # For each step in the trajectory, construct indices for the next n actions
            action_chunk_indices = tf.broadcast_to(
                tf.range(action_chunk_size)[None],
                [traj_len, action_chunk_size],
            ) + tf.broadcast_to(
                tf.range(traj_len)[:, None],
                [traj_len, action_chunk_size],
            )

            # Cap to length of the sequence --> final chunks will repeat the last action
            # This makes sense, since we are using absolute joint + gripper position actions
            action_chunk_indices = tf.minimum(
                action_chunk_indices, traj_len - 1)

            # Gather the actions for each chunk
            traj["actions"] = tf.gather(traj["actions"], action_chunk_indices)
            return traj

        def filter_idle(traj):
            """Filter out chunks with idle actions.
            --> we filter if at least first half of chunk does not move.
            """
            if action_space == TruthActionSpace.JOINT_POSITION:
                # Compute delta to first position in action chunk
                return tf.reduce_any(
                    tf.abs(
                        traj["actions"][: action_chunk_size // 2] -
                        traj["actions"][:1]
                    )
                    > 1e-3
                )
            return tf.reduce_any(
                tf.abs(traj["actions"][: action_chunk_size // 2]) > 1e-3
            )

        # Decode images: RLDS saves encoded images, only decode now for efficiency
        def decode_images(traj):
            traj["observation"]["image"] = tf.io.decode_image(
                traj["observation"]["image"], expand_animations=False, dtype=tf.uint8
            )
            traj["observation"]["wrist_image"] = tf.io.decode_image(
                traj["observation"]["wrist_image"],
                expand_animations=False,
                dtype=tf.uint8,
            )
            return traj

        if chunkflow_stride is None:
            dataset = dataset.traj_map(chunk_actions, num_parallel_calls)
            dataset = dataset.filter(filter_idle)
            dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)
            dataset = dataset.frame_map(decode_images, num_parallel_calls)
        else:
            stride = chunkflow_stride

            def pair_chunks(traj):
                """Build full adjacent windows while the episode axis still exists."""

                traj_len = tf.shape(traj["actions"])[0]
                pair_count = tf.maximum(traj_len - stride - action_chunk_size + 1, 0)
                starts = tf.range(0, pair_count, delta=stride)
                offsets = tf.range(action_chunk_size)[None, :]
                previous_indices = starts[:, None] + offsets
                current_indices = starts[:, None] + stride + offsets

                previous_actions = tf.gather(traj["actions"], previous_indices)
                current_actions = tf.gather(traj["actions"], current_indices)

                def gather_history(starts_to_condition):
                    indices = starts_to_condition[:, None] - history_length + tf.range(history_length)[None, :]
                    mask = indices >= 0
                    values = tf.gather(traj["actions"], tf.maximum(indices, 0))
                    values = tf.where(mask[..., None], values, tf.zeros_like(values))
                    return values, mask

                previous_history, previous_history_mask = gather_history(starts)
                current_history, current_history_mask = gather_history(starts + stride)

                def observation_at(indices):
                    return {
                        "image": tf.gather(traj["observation"]["image"], indices),
                        "wrist_image": tf.gather(traj["observation"]["wrist_image"], indices),
                        "joint_position": tf.gather(traj["observation"]["joint_position"], indices),
                    }

                previous = {
                    "actions": previous_actions,
                    "observation": observation_at(starts),
                    "prompt": tf.gather(traj["prompt"], starts),
                    "action_history": previous_history,
                    "action_history_mask": previous_history_mask,
                }
                current = {
                    "actions": current_actions,
                    "observation": observation_at(starts + stride),
                    "prompt": tf.gather(traj["prompt"], starts + stride),
                    "action_history": current_history,
                    "action_history_mask": current_history_mask,
                }
                return {"previous": previous, "current": current}

            def filter_idle_pair(pair):
                actions = pair["current"]["actions"]
                if action_space == TruthActionSpace.JOINT_POSITION:
                    return tf.reduce_any(
                        tf.abs(actions[: action_chunk_size // 2] - actions[:1]) > 1e-3
                    )
                return tf.reduce_any(tf.abs(actions[: action_chunk_size // 2]) > 1e-3)

            def decode_pair_images(pair):
                for key in ("previous", "current"):
                    pair[key]["observation"]["image"] = tf.io.decode_image(
                        pair[key]["observation"]["image"],
                        expand_animations=False,
                        dtype=tf.uint8,
                    )
                    pair[key]["observation"]["wrist_image"] = tf.io.decode_image(
                        pair[key]["observation"]["wrist_image"],
                        expand_animations=False,
                        dtype=tf.uint8,
                    )
                return pair

            dataset = dataset.traj_map(pair_chunks, num_parallel_calls)
            dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)
            dataset = dataset.filter(filter_idle_pair)
            dataset = dataset.frame_map(decode_pair_images, num_parallel_calls)

        # Shuffle, batch
        dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.batch(batch_size)
        # Note =>> Seems to reduce memory usage without affecting speed?
        dataset = dataset.with_ram_budget(1)

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        yield from self.dataset.as_numpy_iterator()

    def __len__(self):
        # This is the approximate number of samples in DROID after filtering.
        # Easier to hardcode than to iterate through the dataset and compute it.
        return 20_000_000
