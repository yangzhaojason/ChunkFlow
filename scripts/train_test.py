import dataclasses
import functools
import pathlib
from typing import NamedTuple

from flax import nnx
import jax
import jax.numpy as jnp
import optax
import pytest

from openpi.models import chunkflow_critic
from openpi.models import model as _model
from openpi.training import checkpoints
from openpi.training import chunkflow_batch
from openpi.training import config as _config
from openpi.training import sharding
from openpi.training import utils as training_utils
from scripts import train


class _ToyFlowOutput(NamedTuple):
    loss: jax.Array
    velocity: jax.Array
    x_t: jax.Array
    time: jax.Array


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

    def encode_critic_state(self, rng, observation, *, train):
        del rng, train
        state = observation.state
        feature = jnp.concatenate((state, state + 1.0, state - 1.0), axis=-1)
        return jax.lax.stop_gradient(feature)

    def first_action_flow_forward(
        self,
        rng,
        observation,
        executed_action,
        *,
        train,
        dummy_tail,
        noise,
        time,
    ):
        del rng, observation, train, dummy_tail
        velocity = self.weight.value * jnp.ones_like(executed_action)
        target_velocity = noise[:, 0] - executed_action
        loss = jnp.mean(jnp.square(velocity - target_velocity), axis=-1)
        x_t = time[:, None] * noise[:, 0] + (1.0 - time[:, None]) * executed_action
        return _ToyFlowOutput(loss=loss, velocity=velocity, x_t=x_t, time=time)

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
    awac_gamma: float = 0.9
    awac_expectile_tau_e: float = 0.7
    awac_temperature_tau: float = 0.5
    awac_wmax: float = 10.0
    awac_actor_weight: float = 1.0
    awac_q_weight: float = 1.0
    awac_v_weight: float = 1.0
    awac_target_decay: float = 0.5
    kl_beta: float = 0.1

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
        action_history=jnp.zeros((batch_size, 1, 1), dtype=jnp.float32),
        action_history_mask=jnp.ones((batch_size, 1), dtype=jnp.bool_),
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


def _tiny_awac_config(*, awac_enable: bool, freeze_filter=nnx.Param) -> _config.TrainConfig:
    return _config.TrainConfig(
        name="toy-awac",
        exp_name="toy",
        model=_ToyModelConfig(awac_enable=awac_enable),
        weight_loader=_ConstantWeightLoader(7.0),
        freeze_filter=freeze_filter,
    )


def _single_device_mesh() -> jax.sharding.Mesh:
    return sharding.make_mesh(1)


def _tiny_composite_batch() -> chunkflow_batch.ChunkFlowTrainBatch:
    observation = _observation()
    return chunkflow_batch.ChunkFlowTrainBatch(
        supervised=chunkflow_batch.PairedChunkBatch(
            previous_observation=_observation(),
            previous_actions=jnp.full((2, 2, 1), 0.25),
            observation=observation,
            actions=jnp.ones((2, 2, 1)),
            step=jnp.zeros((2,), dtype=jnp.int32),
        ),
        transition=chunkflow_batch.StepTransitionBatch(
            observation=observation,
            executed_action=jnp.asarray([[0.2], [-0.1]], dtype=jnp.float32),
            reward=jnp.asarray([0.5, -0.25], dtype=jnp.float32),
            continuation=jnp.asarray([1.0, 0.0], dtype=jnp.float32),
            next_observation=_model.Observation(
                images={},
                image_masks={},
                state=jnp.asarray([[0.3], [0.1]], dtype=jnp.float32),
                action_history=jnp.zeros((2, 1, 1), dtype=jnp.float32),
                action_history_mask=jnp.ones((2, 1), dtype=jnp.bool_),
            ),
            episode_id=jnp.asarray([2, 2], dtype=jnp.int32),
            frame_index=jnp.asarray([4, 5], dtype=jnp.int32),
        ),
    )


def _tiny_awac_state(config: _config.TrainConfig) -> training_utils.TrainState:
    model = _ToyModel()
    params = nnx.state(model)
    critic = chunkflow_critic.ChunkFlowCritic(
        state_dim=model.critic_state_dim,
        action_dim=config.model.action_dim,
        hidden_width=config.model.awac_critic_hidden_width,
        hidden_depth=config.model.awac_critic_depth,
        rngs=nnx.Rngs(jax.random.key(3)),
    )
    target_v_model_def = nnx.graphdef(critic.v)
    target_v_params = jax.tree.map(lambda value: value, nnx.state(critic.v))
    critic_model_def, critic_params = nnx.split(critic)
    tx = optax.adam(0.03)
    return training_utils.TrainState(
        step=jnp.asarray(0, dtype=jnp.int32),
        params=params,
        model_def=nnx.graphdef(model),
        opt_state=tx.init(params.filter(config.trainable_filter)),
        tx=tx,
        ema_decay=0.9,
        ema_params=jax.tree.map(lambda value: value, params),
        critic_params=critic_params,
        critic_model_def=critic_model_def,
        critic_opt_state=tx.init(critic_params),
        target_v_params=target_v_params,
        target_v_model_def=target_v_model_def,
        reference_params=jax.tree.map(lambda value: value, params),
    )


def _tree_equal(actual, expected) -> bool:
    if jax.tree.structure(actual) != jax.tree.structure(expected):
        return False
    return all(
        bool(jnp.array_equal(actual_leaf, expected_leaf))
        for actual_leaf, expected_leaf in zip(
            jax.tree.leaves(actual),
            jax.tree.leaves(expected),
            strict=True,
        )
    )


def _assert_state_allclose(actual: nnx.State, expected: nnx.State) -> None:
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(actual),
        jax.tree.leaves(expected),
        strict=True,
    ):
        assert jnp.allclose(actual_leaf, expected_leaf)


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


def test_jitted_awac_train_step_updates_actor_critic_and_both_emas():
    config = _tiny_awac_config(awac_enable=True, freeze_filter=nnx.Nothing)
    state = _tiny_awac_state(config)
    batch = _tiny_composite_batch()
    before_reference = state.reference_params
    before_target = state.target_v_params

    new_state, metrics = jax.jit(functools.partial(train.train_step, config))(
        jax.random.key(1), state, batch
    )

    assert new_state.step == state.step + 1
    assert not _tree_equal(new_state.params, state.params)
    assert not _tree_equal(new_state.critic_params, state.critic_params)
    assert not _tree_equal(new_state.opt_state, state.opt_state)
    assert not _tree_equal(new_state.critic_opt_state, state.critic_opt_state)
    assert not _tree_equal(new_state.target_v_params, before_target)
    assert _tree_equal(new_state.reference_params, before_reference)

    updated_critic = nnx.merge(new_state.critic_model_def, new_state.critic_params)
    expected_target = jax.tree.map(
        lambda old, online: config.model.awac_target_decay * old
        + (1.0 - config.model.awac_target_decay) * online,
        before_target,
        nnx.state(updated_critic.v),
    )
    _assert_state_allclose(new_state.target_v_params, expected_target)
    expected_actor_ema = jax.tree.map(
        lambda old, current: state.ema_decay * old + (1.0 - state.ema_decay) * current,
        state.ema_params,
        new_state.params,
    )
    _assert_state_allclose(new_state.ema_params, expected_actor_ema)

    assert jnp.allclose(metrics["loss"], metrics["actor_objective"] + metrics["critic_objective"])
    for name in (
        "loss",
        "actor_objective",
        "critic_objective",
        "awac/actor_loss",
        "awac/q_loss",
        "awac/v_loss",
        "awac/reference_consistency",
        "actor_grad_norm",
        "critic_grad_norm",
    ):
        assert jnp.isfinite(metrics[name])


def test_train_step_rejects_awac_config_without_composite_batch():
    config = _tiny_awac_config(awac_enable=True, freeze_filter=nnx.Nothing)

    with pytest.raises(
        ValueError,
        match="ChunkFlow AWAC requires a composite supervised/transition batch",
    ):
        train.train_step(
            config,
            jax.random.key(0),
            _tiny_awac_state(config),
            _tiny_composite_batch().supervised,
        )


def test_train_step_rejects_composite_batch_when_awac_is_disabled():
    config = _tiny_awac_config(awac_enable=False, freeze_filter=nnx.Nothing)

    with pytest.raises(ValueError, match="AWAC is disabled"):
        train.train_step(config, jax.random.key(0), _state(config), _tiny_composite_batch())


@pytest.mark.parametrize(
    "field_name",
    [
        "critic_params",
        "critic_model_def",
        "critic_opt_state",
        "target_v_params",
        "target_v_model_def",
        "reference_params",
    ],
)
def test_train_step_names_missing_awac_state_subtree(field_name):
    config = _tiny_awac_config(awac_enable=True, freeze_filter=nnx.Nothing)
    incomplete_state = dataclasses.replace(_tiny_awac_state(config), **{field_name: None})

    with pytest.raises(ValueError, match=field_name):
        train.train_step(config, jax.random.key(0), incomplete_state, _tiny_composite_batch())


def test_train_step_requires_actor_ema_for_awac_history():
    config = _tiny_awac_config(awac_enable=True, freeze_filter=nnx.Nothing)
    state = dataclasses.replace(_tiny_awac_state(config), ema_params=None)

    with pytest.raises(ValueError, match="ema_params"):
        train.train_step(config, jax.random.key(0), state, _tiny_composite_batch())


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
