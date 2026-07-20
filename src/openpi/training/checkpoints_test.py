import dataclasses

from flax import nnx
import jax
import jax.numpy as jnp
import optax
import pytest

from openpi.models import chunkflow_critic
from openpi.models import model as _model
from openpi.training import checkpoints
from openpi.training import utils as training_utils

_AWAC_STATE_FIELDS = (
    "critic_params",
    "critic_model_def",
    "critic_opt_state",
    "target_v_params",
    "target_v_model_def",
    "reference_params",
)


class _ScalarModel(_model.BaseModel):
    def __init__(self, value: float = 1.0):
        super().__init__(action_dim=1, action_horizon=1, max_token_len=1)
        self.weight = nnx.Param(jnp.asarray(value, dtype=jnp.float32))

    def compute_loss(self, rng, observation, actions, *, train=False):
        del rng, observation, train
        return jnp.square(self.weight.value * actions)

    def sample_actions(self, rng, observation, **kwargs):
        del rng, observation, kwargs
        return jnp.zeros((1, 1, 1), dtype=jnp.float32)


@dataclasses.dataclass(frozen=True)
class _NoAssetsDataConfig:
    norm_stats: None = None
    asset_id: None = None


class _NoAssetsDataLoader:
    def data_config(self):
        return _NoAssetsDataConfig()


def _supervised_train_state() -> training_utils.TrainState:
    model = _ScalarModel()
    params = nnx.state(model)
    tx = optax.sgd(0.1)
    return training_utils.TrainState(
        step=jnp.asarray(0, dtype=jnp.int32),
        params=params,
        model_def=nnx.graphdef(model),
        opt_state=tx.init(params),
        tx=tx,
        ema_decay=0.9,
        ema_params=nnx.state(_ScalarModel(2.0)),
    )


def _awac_train_state() -> training_utils.TrainState:
    state = _supervised_train_state()
    critic = chunkflow_critic.ChunkFlowCritic(
        state_dim=2,
        action_dim=1,
        hidden_width=2,
        hidden_depth=1,
        rngs=nnx.Rngs(0),
    )
    target_v_model_def = nnx.graphdef(critic.v)
    target_v_params = jax.tree.map(lambda value: value, nnx.state(critic.v))
    critic_model_def, critic_params = nnx.split(critic)
    return dataclasses.replace(
        state,
        critic_params=critic_params,
        critic_model_def=critic_model_def,
        critic_opt_state=state.tx.init(critic_params),
        target_v_params=target_v_params,
        target_v_model_def=target_v_model_def,
        reference_params=jax.tree.map(lambda value: value, state.params),
    )


def _save_test_state(checkpoint_dir, state: training_utils.TrainState) -> None:
    manager, resuming = checkpoints.initialize_checkpoint_dir(
        checkpoint_dir,
        keep_period=None,
        overwrite=False,
        resume=False,
    )
    assert not resuming
    try:
        checkpoints.save_state(manager, state, _NoAssetsDataLoader(), step=1)
        manager.wait_until_finished()
    finally:
        manager.close()


def _restore_as_awac(checkpoint_dir) -> training_utils.TrainState:
    manager, resuming = checkpoints.initialize_checkpoint_dir(
        checkpoint_dir,
        keep_period=None,
        overwrite=False,
        resume=True,
    )
    assert resuming
    try:
        return checkpoints.restore_state(manager, _awac_train_state(), _NoAssetsDataLoader())
    finally:
        manager.close()


def test_checkpoint_split_exports_actor_only_and_keeps_awac_training_state():
    state = _awac_train_state()

    training_part, exported = checkpoints._split_params(state)  # noqa: SLF001

    assert exported is state.ema_params
    assert training_part.critic_params is state.critic_params
    assert training_part.critic_model_def is state.critic_model_def
    assert training_part.critic_opt_state is state.critic_opt_state
    assert training_part.target_v_params is state.target_v_params
    assert training_part.target_v_model_def is state.target_v_model_def
    assert training_part.reference_params is state.reference_params
    assert "critic" not in exported.to_pure_dict()

    merged = checkpoints._merge_params(training_part, {"params": exported})  # noqa: SLF001
    assert merged.params is state.params
    assert merged.ema_params is exported
    for field in _AWAC_STATE_FIELDS:
        assert getattr(merged, field) is getattr(state, field)


def test_awac_resume_rejects_checkpoint_without_critic_state(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    _save_test_state(checkpoint_dir, _supervised_train_state())

    with pytest.raises((ValueError, KeyError), match=r"critic|AWAC"):
        _restore_as_awac(checkpoint_dir)


def test_validate_awac_state_names_all_missing_fields_when_enabled():
    state = _supervised_train_state()

    with pytest.raises(ValueError, match="missing") as exc_info:
        checkpoints.validate_awac_state(state, awac_enabled=True)

    for field in _AWAC_STATE_FIELDS:
        assert field in str(exc_info.value)


def test_validate_awac_state_names_all_unexpected_fields_when_disabled():
    state = _awac_train_state()

    with pytest.raises(ValueError, match="unexpected") as exc_info:
        checkpoints.validate_awac_state(state, awac_enabled=False)

    for field in _AWAC_STATE_FIELDS:
        assert field in str(exc_info.value)


def test_validate_awac_state_accepts_complete_or_disabled_state():
    checkpoints.validate_awac_state(_awac_train_state(), awac_enabled=True)
    checkpoints.validate_awac_state(_supervised_train_state(), awac_enabled=False)
