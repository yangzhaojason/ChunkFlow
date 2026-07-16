"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import os
# 在导入其他模块之前设置环境变量
# os.environ["TF_NUM_INTEROP_THREADS"] = "2"
# os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
# os.environ["OMP_NUM_THREADS"] = "2"

import numpy as np
import tqdm
import tyro
import tensorflow as tf

# 配置 TensorFlow 以减少线程使用
# tf.config.threading.set_inter_op_parallelism_threads(2)
# tf.config.threading.set_intra_op_parallelism_threads(2)

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms

# 完全禁用 GPU，强制使用 CPU
# tf.config.set_visible_devices([], "GPU")

# # Set environment variables to avoid video processing issues
# os.environ["LEROBOT_SKIP_VIDEO"] = "1"
# os.environ["DISABLE_TORCHCODEC"] = "1"

# print("intra_op threads:", tf.config.threading.get_intra_op_parallelism_threads())
# print("inter_op threads:", tf.config.threading.get_inter_op_parallelism_threads())

class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {
            k: v
            for k, v in x.items()
            if not np.issubdtype(np.asarray(v).dtype, np.str_)
        }


def create_rlds_truth_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_truth_rlds_dataset_cartesian(
        data_config, action_horizon, batch_size, shuffle=False
    )
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size

    else:
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    # print(num_batches)
    # exit(0)
    return data_loader, num_batches


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    print(data_config.rlds_data_dir)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_truth_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        raise ValueError("RLDS data dir is not set")

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            values = np.asarray(batch[key][0])
            stats[key].update(values.reshape(-1, values.shape[-1]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
