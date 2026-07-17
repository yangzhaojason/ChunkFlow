from types import SimpleNamespace

import pytest

from openpi.policies import policy_config


def test_pytorch_checkpoint_rejects_unsupported_chunkflow_history(tmp_path):
    (tmp_path / "model.safetensors").touch()
    config = SimpleNamespace(model=SimpleNamespace(history_length=2))

    with pytest.raises(NotImplementedError, match=r"PyTorch.*history"):
        policy_config.create_trained_policy(config, tmp_path)
