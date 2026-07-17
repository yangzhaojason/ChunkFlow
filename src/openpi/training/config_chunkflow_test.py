import pathlib

import pytest

from openpi.models import pi0_config
from openpi.training import config as _config


def test_libero_rejects_episode_success_reward_synthesis():
    factory = _config.LeRobotLiberoDataConfig(success_map_path="legacy-success-map.json")

    with pytest.raises(ValueError, match="step-aligned rewards and discounts"):
        factory.create(pathlib.Path("/tmp/assets"), pi0_config.Pi0Config())
