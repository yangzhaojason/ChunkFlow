"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.

Usage:
    uv run scripts/compute_norm_stats_new.py <config_name> [--max-frames <num>]

Examples:
    uv run scripts/compute_norm_stats_new.py pi0_fast_truth_finetune
    uv run scripts/compute_norm_stats_new.py pi05_truth_finetune_cartesian --max-frames 10000
    uv run scripts/compute_norm_stats_new.py pi05_cytoderm10_joint_arm_move --max-frames 10000
    uv run scripts/compute_norm_stats_new.py pi05_cytoderm11_joint_arm_move --max-frames 10000
    uv run scripts/compute_norm_stats_new.py pi05_cytoderm11_joint_arm_move_chunkflow --max-frames 100000
    uv run scripts/compute_norm_stats_new.py pi05_cytoderm13_joint_arm_move_chunkflow --max-frames 100000
"""

from typing import Annotated

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    """Remove string fields from the data since they are not supported by JAX."""

    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


# Dataset types that need special batch processing (reshape for dual-arm or special data format)
# Note: These datasets have nested structure where batch[key] returns data with shape
# (batch_size, arms/chunks, ...) and we need batch[key][0] to extract the actual data
SPECIAL_BATCH_PROCESSING_DATASETS = {
    _data_loader.TruthRldsDatasetCartesian,
    # _data_loader.TruthRldsDatasetDualCartesian,
    _data_loader.TruthRldsDatasetJointWithoutGripper,
    # _data_loader.FrankaRldsDataset,
}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.TorchDataLoader, int]:
    """Create a PyTorch-based data loader for computing norm stats."""
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")

    dataset = _data_loader.create_torch_dataset(
        data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
    )

    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False

    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    config_name: str,
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.RLDSDataLoader, int, bool]:
    """Create an RLDS data loader for computing norm stats.

    Returns:
        data_loader: The RLDS data loader
        num_batches: Number of batches to process
        needs_special_processing: Whether this dataset needs special batch processing
    """
    # Get the dataset class from config name
    dataset_class = _data_loader.CONFIG_NAME.get(config_name)
    if dataset_class is None:
        raise ValueError(
            f"Unknown config name: {config_name}. "
            f"Available configs: {list(_data_loader.CONFIG_NAME.keys())}"
        )

    # Prepare dataset arguments
    dataset_kwargs = {
        "data_dir": data_config.rlds_data_dir,
        "batch_size": batch_size,
        "shuffle": False,
        "action_chunk_size": action_horizon,
        "action_space": data_config.action_space,
        "filter_dict_path": data_config.filter_dict_path,
    }

    # Add repo_id for datasets that need it (all except DroidRldsNewDataset)
    if dataset_class != _data_loader.DroidRldsNewDataset:
        dataset_kwargs["repo_id"] = data_config.repo_id

    # Add downsampled_and_repeated for datasets that support it
    if dataset_class in (
        _data_loader.TruthRldsDatasetCartesian,
        # _data_loader.TruthRldsDatasetDualCartesian,
    ):
        dataset_kwargs["downsampled_and_repeated"] = data_config.downsampled_and_repeated

    # Create the dataset
    dataset = dataset_class(**dataset_kwargs)

    # Apply transformations
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
        is_batched=True,
    )

    # Calculate number of batches
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        num_batches = len(dataset) // batch_size

    # Create data loader
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )

    # Check if this dataset needs special processing
    needs_special_processing = dataset_class in SPECIAL_BATCH_PROCESSING_DATASETS

    return data_loader, num_batches, needs_special_processing


def main(
    config_name: Annotated[str, tyro.conf.Positional],
    max_frames: int | None = None,
):
    """Compute normalization statistics for a given config.

    Args:
        config_name: Name of the config to use (e.g., "pi0_fast_truth_finetune")
        max_frames: Maximum number of frames to use for computing stats (optional)
    """
    print(f"Computing normalization statistics for config: {config_name}")

    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    print(f"Data config: {data_config.repo_id}")
    if data_config.rlds_data_dir:
        print(f"RLDS data directory: {data_config.rlds_data_dir}")

    # Determine which data loader to use
    if data_config.rlds_data_dir is not None:
        data_loader, num_batches, needs_special_processing = create_rlds_dataloader(
            config_name, data_config, config.model.action_horizon, config.batch_size, max_frames
        )
        print(
            f"Using RLDS data loader with special processing: {needs_special_processing}")
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            config.model,
            config.num_workers,
            max_frames
        )
        needs_special_processing = False
        print("Using PyTorch data loader")

    # Initialize statistics collectors
    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    # Compute statistics
    print(f"Processing {num_batches} batches...")
    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            if needs_special_processing:
                # Special processing for certain RLDS datasets
                # These datasets have shape (batch, arms, ...) and we need to flatten properly
                values = np.asarray(batch[key][0])
                stats[key].update(values.reshape(-1, values.shape[-1]))
            else:
                # Standard processing
                stats[key].update(np.asarray(batch[key]))

    # Get final statistics
    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    # Save statistics
    output_path = config.assets_dirs / data_config.repo_id
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nStatistics computed:")
    for key, stat in norm_stats.items():
        print(f"  {key}:")
        print(f"    mean shape: {stat.mean.shape}")
        print(f"    std shape: {stat.std.shape}")

    print(f"\nWriting statistics to: {output_path}")
    normalize.save(output_path, norm_stats)
    print("Done!")


if __name__ == "__main__":
    tyro.cli(main)
