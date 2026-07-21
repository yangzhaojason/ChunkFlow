import dataclasses
import pathlib

import pytest

from openpi.training import config as _config
from openpi.training import data_loader
from scripts import train_pytorch


def test_train_loop_rejects_awac_before_top_level_side_effects(monkeypatch, tmp_path):
    config = _config.get_config("debug_pi05")
    config = dataclasses.replace(
        config,
        model=dataclasses.replace(config.model, awac_enable=True),
        checkpoint_base_dir=str(tmp_path),
        exp_name="awac-guard",
        overwrite=True,
        resume=False,
    )
    config.checkpoint_dir.mkdir(parents=True)
    marker = config.checkpoint_dir / "keep-me.txt"
    marker.write_text("untouched")
    calls = []

    def fail_if_called(name):
        def fail(*args, **kwargs):
            del args, kwargs
            calls.append(name)
            raise AssertionError(f"{name} must not run before the AWAC PyTorch guard")

        return fail

    monkeypatch.setattr(train_pytorch, "setup_ddp", fail_if_called("setup_ddp"))
    monkeypatch.setattr(train_pytorch.shutil, "rmtree", fail_if_called("checkpoint deletion"))
    monkeypatch.setattr(pathlib.Path, "mkdir", fail_if_called("checkpoint creation"))
    monkeypatch.setattr(train_pytorch, "init_wandb", fail_if_called("wandb init"))
    monkeypatch.setattr(train_pytorch, "build_datasets", fail_if_called("loader construction"))

    with pytest.raises(NotImplementedError) as exc_info:
        train_pytorch.train_loop(config)

    assert str(exc_info.value) == "ChunkFlow AWAC training is supported only by the JAX trainer"
    assert calls == []
    assert marker.read_text() == "untouched"


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
