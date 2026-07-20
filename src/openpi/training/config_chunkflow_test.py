import os
import pathlib

import pytest

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import weight_loaders


def test_libero_rejects_episode_success_reward_synthesis():
    factory = _config.LeRobotLiberoDataConfig(success_map_path="legacy-success-map.json")

    with pytest.raises(ValueError, match="step-aligned rewards and discounts"):
        factory.create(pathlib.Path("/tmp/assets"), pi0_config.Pi0Config())


def test_paper_chunkflow_environment_controls_follow_import_environment():
    assert os.environ.get(
        "CHUNKFLOW_PAPER_REPO_ID", "chunkflow/libero"
    ) == _config.CHUNKFLOW_PAPER_REPO_ID
    assert os.environ.get(
        "CHUNKFLOW_PAPER_DATASET_ROOT", "datasets/chunkflow_lerobot"
    ) == _config.CHUNKFLOW_PAPER_DATASET_ROOT
    assert os.environ.get(
        "CHUNKFLOW_SUPERVISED_CHECKPOINT",
        "checkpoints/pi05_chunkflow_paper_bc/chunkflow_bc/29999/params",
    ) == _config.CHUNKFLOW_SUPERVISED_CHECKPOINT


@pytest.mark.parametrize(
    ("name", "awac_enable", "checkpoint_attr"),
    [
        ("pi05_chunkflow_paper_bc", False, "CHUNKFLOW_PI05_BASE_CHECKPOINT"),
        ("pi05_chunkflow_paper_awac", True, "CHUNKFLOW_SUPERVISED_CHECKPOINT"),
    ],
)
def test_paper_chunkflow_configs_use_public_contract(name, awac_enable, checkpoint_attr):
    config = _config.get_config(name)
    libero = _config.get_config("pi05_libero")

    assert isinstance(config.model, pi0_config.Pi0Config)
    assert config.model.pi05
    assert config.model.action_dim == 32
    assert config.model.action_horizon == 10
    assert config.model.overlap_O == 8
    assert config.model.chunk_stride == 2
    assert config.model.history_length == 4
    assert config.model.continuity_first_order_weight == pytest.approx(0.005)
    assert config.model.continuity_second_order_weight == pytest.approx(0.005)
    assert config.model.boundary_weight == pytest.approx(0.03)
    assert config.model.history_noise_std == 0.0
    assert config.model.history_dropout_probability == 0.0
    assert config.model.history_schedule_max_alpha == 0.0
    assert config.model.awac_enable is awac_enable

    assert config.batch_size == 256
    assert config.num_train_steps == 30_000
    assert config.ema_decay == pytest.approx(0.999)
    assert config.lr_schedule == libero.lr_schedule
    assert config.optimizer == libero.optimizer
    assert isinstance(config.weight_loader, weight_loaders.CheckpointWeightLoader)
    assert config.weight_loader.params_path == getattr(_config, checkpoint_attr)

    assert isinstance(config.data, _config.LeRobotLiberoDataConfig)
    assert config.data.repo_id == _config.CHUNKFLOW_PAPER_REPO_ID
    assert config.data.lerobot_root == _config.CHUNKFLOW_PAPER_DATASET_ROOT
    assert config.data.extra_delta_transform is False
    assert config.data.base_config is not None
    assert config.data.base_config.prompt_from_task is True
    assert config.data.base_config.awac_executed_action_key == "actions"
    assert config.data.base_config.awac_reward_key == "rewards"
    assert config.data.base_config.awac_continuation_key == "discounts"


def test_paper_chunkflow_awac_config_uses_equation_defaults():
    model = _config.get_config("pi05_chunkflow_paper_awac").model

    assert model.awac_gamma == pytest.approx(0.99)
    assert model.awac_expectile_tau_e == pytest.approx(0.7)
    assert model.awac_temperature_tau == pytest.approx(0.05)
    assert model.awac_wmax == pytest.approx(20.0)
    assert model.kl_beta == pytest.approx(2e-4)
