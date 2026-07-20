import dataclasses
import functools
import pathlib

from flax import nnx
import jax
import jax.numpy as jnp
import optax
import pytest

from openpi.models import model as _model
from openpi.training import checkpoints
from openpi.training import chunkflow_batch
from openpi.training import config as _config
from openpi.training import sharding
from openpi.training import utils as training_utils
from scripts import train


class _ToyModel(_model.BaseModel):
    def __init__(self, rngs=None):
        del rngs
        super().__init__(action_dim=1, action_horizon=2, max_token_len=1)
        self.critic_state_dim = 3
        self.weight = nnx.Param(jnp.array(1.0))

    def compute_loss(self, rng, observation, actions, *, train=False):
        del rng, observation, train
        return jnp.square(self.weight.value * actions)

    def compute_paired_loss(
        self,
        rng,
        previous_observation,
        previous_actions,
        observation,
        actions,
        *,
        ema_model,
        step,
        train,
    ):
        del rng, previous_observation, previous_actions, observation, train
        ema_anchor = jax.lax.stop_gradient(ema_model.weight.value)
        loss = jnp.mean(jnp.square(self.weight.value * actions)) + 0.0 * ema_anchor + 0.0 * step
        return loss, {"paired/step": step}

    def sample_actions(self, rng, observation, **kwargs):
        del rng, observation, kwargs
        return jnp.zeros((1, 2, 1))


@dataclasses.dataclass(frozen=True)
class _ToyModelConfig(_model.BaseModelConfig):
    action_dim: int = 1
    action_horizon: int = 2
    max_token_len: int = 1
    history_length: int = 1
    chunkflow_supervised_enabled: bool = True
    awac_enable: bool = False
    awac_critic_hidden_width: int = 4
    awac_critic_depth: int = 1

    @property
    def model_type(self):
        return _model.ModelType.PI0

    def create(self, rng):
        del rng
        return _ToyModel()

    def inputs_spec(self, *, batch_size=1):
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class _ConstantWeightLoader:
    value: float

    def load(self, params):
        return jax.tree.map(
            lambda spec: jnp.full(spec.shape, self.value, dtype=spec.dtype),
            params,
        )


def _observation(batch_size=2):
    return _model.Observation(
        images={},
        image_masks={},
        state=jnp.zeros((batch_size, 1)),
    )


def _state(config):
    model = _ToyModel()
    params = nnx.state(model)
    tx = optax.sgd(0.1)
    return training_utils.TrainState(
        step=jnp.array(0, dtype=jnp.int32),
        params=params,
        model_def=nnx.graphdef(model),
        opt_state=tx.init(params.filter(config.trainable_filter)),
        tx=tx,
        ema_decay=0.9,
        ema_params=params,
    )


def _assert_state_equal(actual: nnx.State, expected: nnx.State) -> None:
    actual_dict = actual.to_pure_dict()
    expected_dict = expected.to_pure_dict()
    assert jax.tree.structure(actual_dict) == jax.tree.structure(expected_dict)
    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(actual_dict),
        jax.tree.leaves(expected_dict),
        strict=True,
    ):
        assert jnp.array_equal(actual_leaf, expected_leaf)


def _tiny_awac_config(*, awac_enable: bool) -> _config.TrainConfig:
    return _config.TrainConfig(
        name="toy-awac",
        exp_name="toy",
        model=_ToyModelConfig(awac_enable=awac_enable),
        weight_loader=_ConstantWeightLoader(7.0),
        freeze_filter=nnx.Param,
    )


def _single_device_mesh() -> jax.sharding.Mesh:
    return sharding.make_mesh(1)


def test_awac_initialization_snapshots_loaded_actor_and_copies_online_value():
    state, _ = train.init_train_state(
        _tiny_awac_config(awac_enable=True),
        jax.random.key(0),
        _single_device_mesh(),
        resume=False,
    )

    assert state.critic_params is not None
    assert state.critic_model_def is not None
    assert state.critic_opt_state is not None
    assert state.target_v_params is not None
    assert state.target_v_model_def is not None
    assert state.reference_params is not None
    _assert_state_equal(state.params, state.reference_params)
    actor_leaves = jax.tree.leaves(state.params.to_pure_dict())
    assert actor_leaves
    assert all(jnp.all(value == 7.0) for value in actor_leaves)
    assert all(value.dtype == jnp.bfloat16 for value in actor_leaves)

    critic = nnx.merge(state.critic_model_def, state.critic_params)
    _assert_state_equal(nnx.state(critic.v), state.target_v_params)


def test_awac_off_initialization_keeps_training_only_state_empty():
    state, _ = train.init_train_state(
        _tiny_awac_config(awac_enable=False),
        jax.random.key(1),
        _single_device_mesh(),
        resume=False,
    )

    assert all(
        getattr(state, field) is None
        for field in (
            "critic_params",
            "critic_model_def",
            "critic_opt_state",
            "target_v_params",
            "target_v_model_def",
            "reference_params",
        )
    )
    checkpoints.validate_awac_state(state, awac_enabled=False)


def test_train_step_consumes_paired_batch_and_replaces_loader_step():
    config = _config.TrainConfig(
        name="toy",
        exp_name="toy",
        model=_ToyModelConfig(),
    )
    batch = chunkflow_batch.PairedChunkBatch(
        previous_observation=_observation(),
        previous_actions=jnp.ones((2, 2, 1)),
        observation=_observation(),
        actions=jnp.ones((2, 2, 1)),
        step=jnp.zeros((2,), dtype=jnp.int32),
    )

    step_fn = jax.jit(functools.partial(train.train_step, config))
    new_state, metrics = step_fn(jax.random.key(0), _state(config), batch)

    assert new_state.step == 1
    assert metrics["paired/step"].shape == ()
    assert metrics["paired/step"] == 0
    assert metrics["loss"] == 1.0


def test_train_step_keeps_legacy_tuple_path():
    config = _config.TrainConfig(
        name="toy-legacy",
        exp_name="toy",
        model=_ToyModelConfig(history_length=0, chunkflow_supervised_enabled=False),
    )

    step_fn = jax.jit(functools.partial(train.train_step, config))
    new_state, metrics = step_fn(
        jax.random.key(0),
        _state(config),
        (_observation(), jnp.ones((2, 2, 1))),
    )

    assert new_state.step == 1
    assert metrics["loss"] == 1.0
    assert "paired/step" not in metrics


def test_train_step_rejects_legacy_batch_when_chunkflow_supervision_is_enabled():
    config = _config.TrainConfig(
        name="toy-misconfigured",
        exp_name="toy",
        model=_ToyModelConfig(),
    )

    step_fn = jax.jit(functools.partial(train.train_step, config))
    with pytest.raises(ValueError, match="requires an episode-paired batch"):
        step_fn(
            jax.random.key(0),
            _state(config),
            (_observation(), jnp.ones((2, 2, 1))),
        )


@pytest.mark.parametrize("config_name", ["debug"])
def test_train_and_resume_smoke(tmp_path: pathlib.Path, config_name: str):
    config = dataclasses.replace(
        _config._CONFIGS_DICT[config_name],  # noqa: SLF001
        batch_size=2,
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="test",
        overwrite=False,
        resume=False,
        num_train_steps=2,
        log_interval=1,
    )
    train.main(config)

    config = dataclasses.replace(config, resume=True, num_train_steps=4)
    train.main(config)
