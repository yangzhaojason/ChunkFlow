import dataclasses

import jax
import numpy as np
import pytest

from openpi.models import pi0_config
from openpi.training import chunkflow_batch
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
import openpi.transforms as _transforms


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


@dataclasses.dataclass(frozen=True)
class _ChunkFlowFakeModelConfig(pi0_config.Pi0Config):
    history_length: int = 1

    @property
    def chunkflow_supervised_enabled(self) -> bool:
        return True


def test_torch_loader_emits_paired_chunk_batch_when_supervision_is_enabled():
    model_config = _ChunkFlowFakeModelConfig(
        action_dim=4,
        action_horizon=4,
        overlap_O=2,
    )
    loader = _data_loader.create_torch_data_loader(
        _config.DataConfig(repo_id="fake"),
        model_config=model_config,
        action_horizon=model_config.action_horizon,
        batch_size=2,
        num_batches=1,
        skip_norm_stats=True,
    )

    batch = next(iter(loader))

    assert isinstance(batch, chunkflow_batch.PairedChunkBatch)
    assert batch.previous_actions.shape == (2, 4, 4)
    assert batch.actions.shape == (2, 4, 4)
    assert batch.observation.action_history.shape == (2, 1, 4)
    np.testing.assert_array_equal(np.asarray(batch.observation.action_history_mask), np.ones((2, 1), dtype=bool))
    assert batch.previous_observation.action_history.shape == (2, 1, 4)
    np.testing.assert_array_equal(
        np.asarray(batch.previous_observation.action_history_mask),
        np.array([[False], [True]]),
    )
    np.testing.assert_array_equal(np.asarray(batch.previous_observation.action_history[0]), np.zeros((1, 4)))


class _SingleBatchedPairDataset:
    def __iter__(self):
        yield {
            "previous": {
                "raw_state": np.array([[7.0]], dtype=np.float32),
                "raw_actions": np.array([[[8.0], [9.0], [11.0]]], dtype=np.float32),
            },
            "current": {
                "raw_state": np.array([[8.0]], dtype=np.float32),
                "raw_actions": np.array([[[10.0], [12.0], [14.0]]], dtype=np.float32),
                "action_history": np.array([[[9.0], [11.0]]], dtype=np.float32),
                "action_history_mask": np.array([[True, True]]),
            },
        }

    def __len__(self):
        return 1


def test_iterable_paired_transform_preserves_raw_history_across_repacking():
    dataset = _data_loader.IterablePairedTransformedDataset(
        _SingleBatchedPairDataset(),
        pre_transforms=[
            _transforms.RepackTransform({"state": "raw_state", "actions": "raw_actions"})
        ],
        transforms=[_transforms.DeltaActions(mask=[True])],
        is_batched=True,
    )

    batch = next(iter(dataset))

    np.testing.assert_array_equal(batch["observation"]["action_history"], [[[1.0], [3.0]]])
    np.testing.assert_array_equal(batch["observation"]["action_history_mask"], [[True, True]])
    np.testing.assert_array_equal(batch["actions"], [[[2.0], [4.0], [6.0]]])
    np.testing.assert_array_equal(batch["previous_actions"], [[[1.0], [2.0], [4.0]]])


def test_rlds_loader_enables_episode_pairing_only_for_capable_dataset(monkeypatch):
    captured = {}

    class _PairCapableRldsDataset:
        supports_chunkflow_pairs = True

        def __init__(self, **kwargs):
            captured.update(kwargs)

        def __iter__(self):
            return iter(())

        def __len__(self):
            return 1

    monkeypatch.setitem(_data_loader.CONFIG_NAME, "paired-test", _PairCapableRldsDataset)

    class _CapturingRldsLoader:
        def __init__(self, dataset, **kwargs):
            captured["wrapped_dataset"] = dataset
            captured["loader_kwargs"] = kwargs

        def __iter__(self):
            return iter(())

    monkeypatch.setattr(_data_loader, "RLDSDataLoader", _CapturingRldsLoader)
    model_config = _ChunkFlowFakeModelConfig(action_dim=4, action_horizon=4, overlap_O=2)

    loader = _data_loader.create_rlds_data_loader(
        config_name="paired-test",
        data_config=_config.DataConfig(repo_id="test", rlds_data_dir="/tmp/test"),
        model_config=model_config,
        action_horizon=4,
        batch_size=2,
        num_batches=1,
        skip_norm_stats=True,
    )

    assert captured["chunkflow_stride"] == 2
    assert captured["history_length"] == 1
    assert isinstance(loader, _data_loader.DataLoaderImpl)
    assert isinstance(captured["wrapped_dataset"], _data_loader.IterablePairedTransformedDataset)


def test_rlds_loader_rejects_silent_legacy_fallback_for_chunkflow(monkeypatch):
    class _LegacyRldsDataset:
        def __init__(self, **kwargs):
            del kwargs

    monkeypatch.setitem(_data_loader.CONFIG_NAME, "legacy-test", _LegacyRldsDataset)

    with pytest.raises(NotImplementedError, match="episode-aware paired chunks"):
        _data_loader.create_rlds_data_loader(
            config_name="legacy-test",
            data_config=_config.DataConfig(repo_id="test", rlds_data_dir="/tmp/test"),
            model_config=_ChunkFlowFakeModelConfig(action_dim=4, action_horizon=4, overlap_O=2),
            action_horizon=4,
            batch_size=2,
            skip_norm_stats=True,
        )


def test_rlds_loader_rejects_an_empty_dataset_without_spinning():
    class _EmptyThenError:
        def __init__(self):
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            if self.iterations > 1:
                raise AssertionError("empty dataset was restarted")
            return iter(())

    loader = _data_loader.RLDSDataLoader(_EmptyThenError(), num_batches=1)

    with pytest.raises(RuntimeError, match="produced no batches"):
        next(iter(loader))
