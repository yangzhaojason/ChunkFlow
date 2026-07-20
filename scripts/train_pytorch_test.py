import dataclasses

import pytest

from openpi.training import config as _config
from openpi.training import data_loader
from scripts import train_pytorch


def test_build_datasets_rejects_awac_before_loader_construction(monkeypatch):
    config = _config.get_config("debug_pi05")
    config = dataclasses.replace(
        config,
        model=dataclasses.replace(config.model, awac_enable=True),
    )
    message = "ChunkFlow AWAC training is supported only by the JAX trainer"

    def fail_loader_construction(*args, **kwargs):
        del args, kwargs
        raise AssertionError("AWAC PyTorch guard must run before loader construction")

    monkeypatch.setattr(data_loader, "create_data_loader", fail_loader_construction)

    with pytest.raises(NotImplementedError) as exc_info:
        train_pytorch.build_datasets(config)

    assert str(exc_info.value) == message
