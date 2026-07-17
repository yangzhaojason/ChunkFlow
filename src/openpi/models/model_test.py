from flax import nnx
import jax
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import pi0_fast
from openpi.shared import download
from openpi.shared import nnx_utils


def _chunkflow_observation_dict():
    images = {
        key: np.zeros((1, 224, 224, 3), dtype=np.uint8)
        for key in _model.IMAGE_KEYS
    }
    return {
        "image": images,
        "image_mask": {key: np.ones((1,), dtype=bool) for key in images},
        "state": np.zeros((1, 2), dtype=np.float32),
        "action_history": np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32),
        "action_history_mask": np.array([[True, False]]),
        "rewards": np.array([[0.0, 1.0]], dtype=np.float32),
        "discounts": np.array([[1.0, 0.0]], dtype=np.float32),
        "executed_actions": np.array([[[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32),
    }


def test_observation_round_trips_chunkflow_fields():
    data = _chunkflow_observation_dict()

    observation = _model.Observation.from_dict(data)
    restored = observation.to_dict()

    for key in (
        "action_history",
        "action_history_mask",
        "rewards",
        "discounts",
        "executed_actions",
    ):
        np.testing.assert_array_equal(restored[key], data[key])


def test_preprocess_observation_preserves_chunkflow_fields():
    observation = _model.Observation.from_dict(_chunkflow_observation_dict())

    processed = _model.preprocess_observation(None, observation)

    for field in (
        "action_history",
        "action_history_mask",
        "rewards",
        "discounts",
        "executed_actions",
    ):
        np.testing.assert_array_equal(getattr(processed, field), getattr(observation, field))


def test_pi0_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_lora_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_fast_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)


def test_pi0_fast_lora_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)

    lora_filter = nnx_utils.PathRegex(".*lora.*")
    model_state = nnx.state(model)

    lora_state_elems = list(model_state.filter(lora_filter))
    assert len(lora_state_elems) > 0


@pytest.mark.manual
def test_model_restore():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    model = config.load(
        _model.restore_params(download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params"))
    )

    loss = model.compute_loss(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = model.sample_actions(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)
