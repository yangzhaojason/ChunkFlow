from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import torch

try:
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
except ModuleNotFoundError:  # Optional unless a LeRobot dataset is requested.
    lerobot_dataset = None

import openpi.models.model as _model
from openpi.training import chunkflow_batch as _chunkflow_batch
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
from openpi.training.droid_rlds_dataset import DroidRldsNewDataset
from openpi.training.truth_rlds_dataset import TruthRldsDataset
from openpi.training.truth_rlds_dataset import TruthRldsDatasetCartesian
from openpi.training.truth_rlds_dataset import TruthRldsDatasetJointWithoutGripper

# from openpi.training.franka_rlds_dataset import FrankaRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)
TrainingBatch = (
    tuple[_model.Observation, _model.Actions]
    | _chunkflow_batch.PairedChunkBatch
    | _chunkflow_batch.StepTransitionBatch
    | _chunkflow_batch.ChunkFlowTrainBatch
)

# Type alias for all supported RLDS dataset types
RLDSDatasetType = (
    DroidRldsDataset
    | DroidRldsNewDataset
    | TruthRldsDataset
    | TruthRldsDatasetCartesian
    # TruthRldsDatasetDualCartesian,
    | TruthRldsDatasetJointWithoutGripper
    # FrankaRldsDataset,
)

# Mapping from config name to RLDS dataset class
CONFIG_NAME: dict[str, type[RLDSDatasetType]] = {
    # Truth RLDS Dataset (Joint space)
    "pi0_fast_truth_finetune": TruthRldsDataset,
    # Truth RLDS Dataset (Cartesian space)
    "pi05_truth_finetune_cartesian": TruthRldsDatasetCartesian,
    "pi05_truth_finetune_cartesian_cloth": TruthRldsDatasetCartesian,
    "pi05_truth_finetune_cartesian_wood": TruthRldsDatasetCartesian,
    "pi05_truth_finetune_cartesian_cloth_downsampled": TruthRldsDatasetCartesian,
    "pi05_truth_finetune_cartesian_cloth_new": TruthRldsDatasetCartesian,
    "pi05_truth_finetune_cartesian_catch_wood_and_move": TruthRldsDatasetCartesian,
    # Truth RLDS Dataset (Dual Cartesian space)
    # "pi05_truth_finetune_darmigo3_004": TruthRldsDatasetDualCartesian,
    # Truth RLDS Dataset (Joint space without gripper)
    "pi05_cytoderm10_joint_arm_move": TruthRldsDatasetJointWithoutGripper,
    "pi05_cytoderm11_joint_arm_move": TruthRldsDatasetJointWithoutGripper,
    "pi05_cytoderm12_joint_arm_move": TruthRldsDatasetJointWithoutGripper,
    "pi05_cytoderm11_joint_arm_move_chunkflow": TruthRldsDatasetJointWithoutGripper,
    "pi05_cytoderm13_joint_arm_move_chunkflow": TruthRldsDatasetJointWithoutGripper,
    # DROID RLDS Dataset
    "pi0_fast_droid_finetune": DroidRldsDataset,
    "pi0_fast_droid_finetune_new": DroidRldsNewDataset,
    # Franka RLDS Dataset
    # "pi05_franka_finetune_joint_wood": FrankaRldsDataset,
}


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError(
            "Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError(
            "Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError(
            "Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError(
            "Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError(
            "Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError(
            "Subclasses of DataLoader should implement __iter__.")


class CompositeDataLoader(DataLoader[_chunkflow_batch.ChunkFlowTrainBatch]):
    """Zip independently sampled, already typed supervised and transition streams."""

    def __init__(
        self,
        data_config: _config.DataConfig,
        supervised_loader: DataLoader[_chunkflow_batch.PairedChunkBatch],
        transition_loader: DataLoader[_chunkflow_batch.StepTransitionBatch],
    ):
        self._data_config = data_config
        self._supervised_loader = supervised_loader
        self._transition_loader = transition_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self) -> Iterator[_chunkflow_batch.ChunkFlowTrainBatch]:
        for supervised, transition in zip(self._supervised_loader, self._transition_loader, strict=False):
            yield _chunkflow_batch.ChunkFlowTrainBatch(
                supervised=supervised,
                transition=transition,
            )


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)

    @property
    def source_dataset(self) -> Dataset:
        return self._dataset


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class IterablePairedTransformedDataset(IterableDataset[dict]):
    """Apply one transform pipeline independently to pre-paired episode records."""

    _OPTIONAL_FIELDS = (
        "action_history",
        "action_history_mask",
        "rewards",
        "discounts",
        "executed_actions",
    )

    def __init__(
        self,
        dataset: IterableDataset,
        *,
        pre_transforms: Sequence[_transforms.DataTransformFn] = (),
        transforms: Sequence[_transforms.DataTransformFn] = (),
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._pre_transform = _transforms.compose(pre_transforms)
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if not self._is_batched:
                yield self._transform_pair(sample)
                continue

            leaves = [leaf for leaf in jax.tree.leaves(sample) if hasattr(leaf, "shape") and leaf.ndim > 0]
            if not leaves:
                raise ValueError("Cannot infer batch size from an empty paired sample")
            batch_size = leaves[0].shape[0]
            individual_samples = [
                jax.tree.map(lambda x: x[i], sample)  # noqa: B023
                for i in range(batch_size)
            ]
            transformed = [self._transform_pair(item) for item in individual_samples]
            yield jax.tree.map(lambda *xs: np.stack(xs, axis=0), *transformed)

    def __len__(self) -> int:
        return len(self._dataset)

    def _transform_pair(self, sample: dict) -> dict:
        if set(sample) != {"previous", "current"}:
            raise ValueError("paired iterable samples must contain exactly 'previous' and 'current'")

        previous = self._transform_record(sample["previous"])
        current = self._transform_record(sample["current"])
        if "actions" not in previous or "actions" not in current:
            raise KeyError("transformed paired records must contain actions")

        return {
            "previous_observation": {key: value for key, value in previous.items() if key != "actions"},
            "previous_actions": previous["actions"],
            "observation": {key: value for key, value in current.items() if key != "actions"},
            "actions": current["actions"],
            "step": np.int32(0),
        }

    def _transform_record(self, record: dict) -> dict:
        optional = {key: record[key] for key in self._OPTIONAL_FIELDS if key in record}
        transformed = dict(self._pre_transform(record))
        transformed.update(optional)
        return self._transform(transformed)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()
        self._awac_enable = bool(getattr(model_config, "awac_enable", False))

    def __getitem__(self, index: SupportsIndex) -> dict:
        record_index = index.__index__()
        rng = jax.random.key(record_index)

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        item = {
            **observation.to_dict(),
            "actions": action,
        }
        if self._awac_enable:
            item.update(
                executed_actions=np.array(action[:1], copy=True),
                rewards=np.zeros((1,), dtype=np.float32),
                discounts=np.array([record_index < self._num_samples - 1], dtype=np.float32),
                episode_index=np.int32(0),
                frame_index=np.int32(record_index),
            )
        return item

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    if lerobot_dataset is None:
        raise ModuleNotFoundError(
            "LeRobot dataset support is unavailable. Install the repository's locked lerobot dependency."
        )

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(
        repo_id,
        root=data_config.lerobot_root,
    )
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        root=data_config.lerobot_root,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(
            dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    pre_transforms, transforms = _input_transforms(
        data_config, skip_norm_stats=skip_norm_stats
    )

    return TransformedDataset(
        dataset,
        [*pre_transforms, *transforms],
    )


def _input_transforms(
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool,
) -> tuple[tuple[_transforms.DataTransformFn, ...], tuple[_transforms.DataTransformFn, ...]]:
    """Split canonical repacking from action-coordinate/model transforms."""

    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return (
        tuple(data_config.repack_transforms.inputs),
        (
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ),
    )


def _episode_frame_arrays(dataset: Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Read stable episode/frame identity before any loader shuffle."""

    raw_dataset = dataset
    while isinstance(raw_dataset, TransformedDataset):
        raw_dataset = raw_dataset.source_dataset

    if isinstance(raw_dataset, FakeDataset):
        size = len(raw_dataset)
        return np.zeros((size,), dtype=np.int64), np.arange(size, dtype=np.int64)

    for holder_name in ("hf_dataset", "dataset"):
        holder = getattr(raw_dataset, holder_name, None)
        if holder is None:
            continue
        try:
            episode_ids = np.asarray(holder["episode_index"])
            frame_indices = np.asarray(holder["frame_index"])
        except (KeyError, TypeError):
            continue
        if episode_ids.ndim == 1 and frame_indices.ndim == 1:
            return episode_ids, frame_indices

    raise ValueError(
        "ChunkFlow paired loading requires episode_index and frame_index arrays on the raw dataset"
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[TrainingBatch]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if (
        data_config.rlds_data_dir is not None
    ):

        return create_rlds_data_loader(
            config_name=config.name,
            data_config=data_config,
            model_config=config.model,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )

    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[TrainingBatch]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)

    def _build_torch_stream_loader(
        stream_dataset,
        *,
        stream_seed: int,
        sampler_seed: int = 0,
    ) -> TorchDataLoader:
        # Use TorchDataLoader for both frameworks. PyTorch DDP owns shuffling through
        # its sampler; JAX divides the global batch across processes.
        sampler = None
        if framework == "pytorch":
            if torch.distributed.is_initialized():
                sampler = torch.utils.data.distributed.DistributedSampler(
                    stream_dataset,
                    num_replicas=torch.distributed.get_world_size(),
                    rank=torch.distributed.get_rank(),
                    shuffle=shuffle,
                    seed=sampler_seed,
                    drop_last=True,
                )
                local_batch_size = batch_size // torch.distributed.get_world_size()
            else:
                local_batch_size = batch_size
        else:
            local_batch_size = batch_size // jax.process_count()

        logging.info(f"local_batch_size: {local_batch_size}")
        return TorchDataLoader(
            stream_dataset,
            local_batch_size=local_batch_size,
            sharding=None if framework == "pytorch" else sharding,
            shuffle=(sampler is None and shuffle),
            sampler=sampler,
            num_batches=num_batches,
            num_workers=num_workers,
            seed=stream_seed,
            framework=framework,
        )

    training_enabled = bool(getattr(model_config, "chunkflow_training_enabled", False))
    awac_enabled = bool(getattr(model_config, "awac_enable", False))
    if not training_enabled:
        legacy_dataset = transform_dataset(
            dataset, data_config, skip_norm_stats=skip_norm_stats
        )
        return DataLoaderImpl(
            data_config,
            _build_torch_stream_loader(legacy_dataset, stream_seed=seed),
        )

    episode_ids, frame_indices = _episode_frame_arrays(dataset)
    stride = int(
        getattr(
            model_config,
            "chunk_stride",
            action_horizon - int(getattr(model_config, "overlap_O", 0)),
        )
    )
    history_length = int(getattr(model_config, "history_length", 0))
    pre_transforms, transforms = _input_transforms(
        data_config, skip_norm_stats=skip_norm_stats
    )
    supervised_dataset = _chunkflow_batch.PairedTransformedDataset(
        dataset,
        episode_ids=episode_ids,
        frame_indices=frame_indices,
        stride=stride,
        history_length=history_length,
        action_horizon=action_horizon,
        pre_transforms=pre_transforms,
        transforms=transforms,
    )
    supervised_loader = DataLoaderImpl(
        data_config,
        _build_torch_stream_loader(
            supervised_dataset,
            stream_seed=seed,
            sampler_seed=seed if awac_enabled else 0,
        ),
    )

    if not awac_enabled:
        return supervised_loader

    for field_name in (
        "awac_executed_action_key",
        "awac_reward_key",
        "awac_continuation_key",
    ):
        if getattr(data_config, field_name) is None:
            raise ValueError(f"{field_name} must be configured when AWAC is enabled")

    transition_dataset = _chunkflow_batch.StepTransitionDataset(
        dataset,
        episode_ids=episode_ids,
        frame_indices=frame_indices,
        history_length=history_length,
        executed_action_key=typing.cast(str, data_config.awac_executed_action_key),
        reward_key=typing.cast(str, data_config.awac_reward_key),
        continuation_key=typing.cast(str, data_config.awac_continuation_key),
        pre_transforms=pre_transforms,
        transforms=transforms,
    )
    transition_loader = DataLoaderImpl(
        data_config,
        _build_torch_stream_loader(
            transition_dataset,
            stream_seed=seed + 1,
            sampler_seed=seed + 1,
        ),
    )
    return CompositeDataLoader(data_config, supervised_loader, transition_loader)


def create_rlds_data_loader(
    config_name: str,
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    model_config: _model.BaseModelConfig | None = None,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[TrainingBatch]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        config_name: The config name to determine which dataset class to use.
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        framework: The framework to use ("jax" or "pytorch").
    """
    if framework == "pytorch":
        raise NotImplementedError(
            "PyTorch RLDS data loader is not supported yet")

    dataset_class = CONFIG_NAME[config_name]
    paired_enabled = bool(getattr(model_config, "chunkflow_supervised_enabled", False))
    if paired_enabled and not bool(getattr(dataset_class, "supports_chunkflow_pairs", False)):
        raise NotImplementedError(
            f"{dataset_class.__name__} does not provide episode-aware paired chunks required by ChunkFlow"
        )

    # 准备基础参数
    dataset_kwargs = {
        "repo_id": data_config.repo_id,
        "data_dir": data_config.rlds_data_dir,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "action_chunk_size": action_horizon,
        "action_space": data_config.action_space,
        "filter_dict_path": data_config.filter_dict_path,
    }
    if paired_enabled:
        dataset_kwargs.update(
            chunkflow_stride=int(
                getattr(
                    model_config,
                    "chunk_stride",
                    action_horizon - int(getattr(model_config, "overlap_O", 0)),
                )
            ),
            history_length=int(getattr(model_config, "history_length", 0)),
        )

    # Add this option only for dataset implementations that support it.
    if dataset_class == TruthRldsDatasetCartesian:
        dataset_kwargs["downsampled_and_repeated"] = data_config.downsampled_and_repeated
        if dataset_class == TruthRldsDatasetCartesian:
            logging.info(
                "data_config.downsampled_and_repeated: %s", data_config.downsampled_and_repeated
            )

    dataset_type_names = {
        TruthRldsDataset: "Truth RLDS",
        TruthRldsDatasetCartesian: "Truth RLDS Cartesian",
        TruthRldsDatasetJointWithoutGripper: "Truth RLDS Joint Without Gripper",
        # TruthRldsDatasetDualCartesian: "Truth RLDS Dual Cartesian",
        DroidRldsDataset: "DROID RLDS",
        DroidRldsNewDataset: "DROID RLDS New",
        # FrankaRldsDataset: "Franka RLDS",
    }
    logging.info("Creating %s data loader", dataset_type_names.get(dataset_class, "RLDS"))

    # 创建数据集
    dataset = dataset_class(**dataset_kwargs)

    # 应用数据转换。配对样本的两个 record 必须独立走同一条坐标变换链。
    if paired_enabled:
        pre_transforms, transforms = _input_transforms(
            data_config, skip_norm_stats=skip_norm_stats
        )
        dataset = IterablePairedTransformedDataset(
            dataset,
            pre_transforms=pre_transforms,
            transforms=transforms,
            is_batched=True,
        )
    else:
        dataset = transform_iterable_dataset(
            dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True
        )

    # 创建数据加载器
    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError(
                "Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(
                f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            # Don't shuffle if using sampler
            shuffle=(sampler is None and shuffle),
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    # We've exhausted the dataset. Create a new iterator and start over.
                    break
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the RLDS data loader to make it compatible with openpi.

    All batching already happens in the RLDS dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: RLDSDatasetType,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError(
                "Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        warned_remainder = False
        while True:
            data_iter = iter(self._dataset)
            yielded_this_pass = False
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    # We've exhausted the dataset. Create a new iterator and start over.
                    if not yielded_this_pass:
                        raise RuntimeError(
                            "RLDS dataset produced no batches; check episode lengths, pairing stride, and filters"
                        ) from None
                    break
                # JAX data-parallel sharding over axis "B" requires the global batch dimension
                # to be divisible by the number of devices. RLDS tf.data pipelines may yield
                # a smaller final batch (no drop_remainder), which would crash here.
                # We truncate to the largest divisible prefix (or skip if too small).
                n_devices = jax.device_count()
                batch0 = None
                for leaf in jax.tree.leaves(batch):
                    if isinstance(leaf, np.ndarray) and leaf.ndim > 0:
                        batch0 = leaf.shape[0]
                        break

                if batch0 is not None and batch0 % n_devices != 0:
                    new0 = (batch0 // n_devices) * n_devices
                    if new0 == 0:
                        if not warned_remainder:
                            logging.warning(
                                "Skipping RLDS batch with size %d which is smaller than device_count=%d.",
                                batch0,
                                n_devices,
                            )
                            warned_remainder = True
                        continue

                    if not warned_remainder:
                        logging.warning(
                            "Truncating RLDS batch from %d to %d to be divisible by device_count=%d.",
                            batch0,
                            new0,
                            n_devices,
                        )
                        warned_remainder = True

                    batch = jax.tree.map(
                        lambda x, new0=new0, batch0=batch0: x[:new0]
                        if isinstance(x, np.ndarray) and x.ndim > 0 and x.shape[0] == batch0
                        else x,
                        batch,
                    )

                num_items += 1
                yielded_this_pass = True
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            if "previous_observation" in batch:
                yield _chunkflow_batch.PairedChunkBatch(
                    previous_observation=_model.Observation.from_dict(batch["previous_observation"]),
                    previous_actions=batch["previous_actions"],
                    observation=_model.Observation.from_dict(batch["observation"]),
                    actions=batch["actions"],
                    step=batch["step"],
                )
            elif "next_observation" in batch:
                yield _chunkflow_batch.StepTransitionBatch(
                    observation=_model.Observation.from_dict(batch["observation"]),
                    executed_action=batch["executed_action"],
                    reward=batch["reward"],
                    continuation=batch["continuation"],
                    next_observation=_model.Observation.from_dict(batch["next_observation"]),
                    episode_id=batch["episode_id"],
                    frame_index=batch["frame_index"],
                )
            else:
                yield _model.Observation.from_dict(batch), batch["actions"]
