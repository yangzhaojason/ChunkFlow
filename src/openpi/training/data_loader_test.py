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


def test_torch_loader_emits_paired_chunk_batch_when_supervision_is_enabled():
    model_config = pi0_config.Pi0Config(
        action_dim=4,
        action_horizon=4,
        overlap_O=2,
        history_length=1,
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


def test_awac_loader_returns_composite_batch_with_independent_streams():
    config = pi0_config.Pi0Config(
        action_dim=4,
        action_horizon=4,
        overlap_O=2,
        history_length=2,
        awac_enable=True,
    )
    data_config = _config.DataConfig(
        repo_id="fake",
        awac_executed_action_key="executed_actions",
        awac_reward_key="rewards",
        awac_continuation_key="discounts",
    )
    loader = _data_loader.create_torch_data_loader(
        data_config,
        model_config=config,
        action_horizon=4,
        batch_size=2,
        num_batches=1,
        skip_norm_stats=True,
        seed=7,
    )

    batch = next(iter(loader))

    assert isinstance(batch, chunkflow_batch.ChunkFlowTrainBatch)
    assert loader.data_config() is data_config
    assert batch.supervised.actions.shape == (2, 4, 4)
    assert batch.transition.executed_action.shape == (2, 4)
    assert batch.transition.reward.shape == (2,)


@pytest.mark.parametrize(
    "missing",
    [
        "awac_executed_action_key",
        "awac_reward_key",
        "awac_continuation_key",
    ],
)
def test_composite_loader_rejects_missing_awac_data_key_early_and_specifically(missing):
    keys = {
        "awac_executed_action_key": "executed_actions",
        "awac_reward_key": "rewards",
        "awac_continuation_key": "discounts",
    }
    keys[missing] = None
    data_config = _config.DataConfig(repo_id="fake", **keys)
    model_config = pi0_config.Pi0Config(action_horizon=4, overlap_O=2, awac_enable=True)

    with pytest.raises(ValueError, match=missing):
        _data_loader.create_torch_data_loader(
            data_config,
            model_config=model_config,
            action_horizon=4,
            batch_size=2,
            num_batches=1,
            skip_norm_stats=True,
        )


def test_composite_loader_uses_independent_seeds_for_supervised_and_transition_streams(monkeypatch):
    seeds = []

    class _CapturingTorchDataLoader:
        def __init__(self, dataset, **kwargs):
            del dataset
            seeds.append(kwargs["seed"])

        def __iter__(self):
            return iter(())

    monkeypatch.setattr(_data_loader, "TorchDataLoader", _CapturingTorchDataLoader)
    model_config = pi0_config.Pi0Config(action_horizon=4, overlap_O=2, awac_enable=True)

    loader = _data_loader.create_torch_data_loader(
        _config.DataConfig(
            repo_id="fake",
            awac_executed_action_key="executed_actions",
            awac_reward_key="rewards",
            awac_continuation_key="discounts",
        ),
        model_config=model_config,
        action_horizon=4,
        batch_size=2,
        num_batches=1,
        skip_norm_stats=True,
        seed=7,
    )

    assert isinstance(loader, _data_loader.CompositeDataLoader)
    assert seeds == [7, 8]


def test_transition_fake_dataset_fields_are_awac_only_and_terminal_discount_is_zero():
    disabled = _data_loader.FakeDataset(pi0_config.Pi0Config(action_horizon=4), 2)[1]
    enabled_dataset = _data_loader.FakeDataset(
        pi0_config.Pi0Config(action_horizon=4, awac_enable=True),
        2,
    )
    first = enabled_dataset[0]
    terminal = enabled_dataset[1]

    for key in ("executed_actions", "rewards", "discounts"):
        assert disabled[key] is None
        assert first[key] is not None
    for key in ("episode_index", "frame_index"):
        assert key not in disabled
        assert key in first
    np.testing.assert_array_equal(first["executed_actions"], first["actions"][:1])
    assert first["discounts"].item() == 1.0
    assert terminal["discounts"].item() == 0.0
    assert terminal["episode_index"] == 0
    assert terminal["frame_index"] == 1


def test_awac_data_config_keys_default_to_none():
    data_config = _config.DataConfig()

    assert data_config.awac_executed_action_key is None
    assert data_config.awac_reward_key is None
    assert data_config.awac_continuation_key is None


def test_legacy_loader_without_chunkflow_or_awac_keeps_tuple_output():
    model_config = pi0_config.Pi0Config(action_dim=4, action_horizon=4)
    loader = _data_loader.create_torch_data_loader(
        _config.DataConfig(repo_id="fake"),
        model_config=model_config,
        action_horizon=4,
        batch_size=2,
        num_batches=1,
        skip_norm_stats=True,
    )

    batch = next(iter(loader))

    assert isinstance(batch, tuple)
    assert len(batch) == 2
    assert batch[1].shape == (2, 4, 4)


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
    model_config = pi0_config.Pi0Config(
        action_dim=4,
        action_horizon=4,
        overlap_O=2,
        history_length=1,
    )

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
            model_config=pi0_config.Pi0Config(
                action_dim=4,
                action_horizon=4,
                overlap_O=2,
                history_length=1,
            ),
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
