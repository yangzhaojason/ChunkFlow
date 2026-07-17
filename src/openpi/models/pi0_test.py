import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from openpi.models import model as _model
from openpi.models.pi0 import Pi0
from openpi.models.pi0 import history_action_masks
from openpi.models.pi0 import make_attn_mask
import openpi.models.pi0_config as _pi0_config


class _IdentityProjection:
    out_features = 2

    def __call__(self, value):
        return value


class _TinyPi05(Pi0):
    def __init__(self):
        self.pi05 = True
        self.action_dim = 2
        self.action_horizon = 2
        self.history_length = 2
        self.action_in_proj = _IdentityProjection()
        self.time_mlp_in = _IdentityProjection()
        self.time_mlp_out = _IdentityProjection()


class _FirstTwoProjection:
    out_features = 2

    def __call__(self, value):
        return value[..., :2]


class _TinyPi0(Pi0):
    def __init__(self):
        self.pi05 = False
        self.action_dim = 2
        self.action_horizon = 2
        self.history_length = 2
        self.state_proj = _IdentityProjection()
        self.action_in_proj = _IdentityProjection()
        self.action_time_mlp_in = _FirstTwoProjection()
        self.action_time_mlp_out = _IdentityProjection()


def _suffix_observation(*, history=True, mask=True):
    return _model.Observation(
        images={},
        image_masks={},
        state=jnp.zeros((1, 2), dtype=jnp.float32),
        action_history=(
            jnp.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=jnp.float32) if history else None
        ),
        action_history_mask=jnp.array([[True, False]]) if mask else None,
    )


def test_embed_suffix_projects_history_before_future_actions():
    model = _TinyPi05()
    noisy_actions = jnp.array([[[10.0, 11.0], [12.0, 13.0]]], dtype=jnp.float32)

    tokens, input_mask, ar_mask, _ = model.embed_suffix(
        _suffix_observation(), noisy_actions, jnp.ones((1,), dtype=jnp.float32)
    )

    assert tokens.shape == (1, 4, 2)
    assert tokens[0, :2].tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert tokens[0, 2:].tolist() == noisy_actions[0].tolist()
    assert input_mask.tolist() == [[True, False, True, True]]
    assert ar_mask.tolist() == [True, False, True, False]


def test_pi0_suffix_places_state_before_history_and_future_actions():
    tokens, input_mask, ar_mask, adarms_cond = _TinyPi0().embed_suffix(
        _suffix_observation(),
        jnp.ones((1, 2, 2), dtype=jnp.float32),
        jnp.ones((1,), dtype=jnp.float32),
    )

    assert tokens.shape == (1, 5, 2)
    assert input_mask.tolist() == [[True, True, False, True, True]]
    assert ar_mask.tolist() == [True, True, False, True, False]
    assert adarms_cond is None


@pytest.mark.parametrize(
    "observation",
    [
        _suffix_observation(history=True, mask=False),
        _suffix_observation(history=False, mask=True),
    ],
)
def test_embed_suffix_requires_history_and_mask_together(observation):
    with pytest.raises(ValueError, match="action_history"):
        _TinyPi05().embed_suffix(
            observation,
            jnp.ones((1, 2, 2), dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.float32),
        )


def test_embed_suffix_keeps_legacy_layout_when_history_is_absent():
    tokens, input_mask, ar_mask, _ = _TinyPi05().embed_suffix(
        _suffix_observation(history=False, mask=False),
        jnp.ones((1, 2, 2), dtype=jnp.float32),
        jnp.ones((1,), dtype=jnp.float32),
    )

    assert tokens.shape == (1, 2, 2)
    assert input_mask.tolist() == [[True, True]]
    assert ar_mask.tolist() == [True, False]


def test_history_and_action_tokens_form_two_causal_segments():
    history_mask = jnp.array([[False, True, True]])

    input_mask, ar_mask = history_action_masks(history_mask, action_horizon=4)

    assert input_mask.tolist() == [[False, True, True, True, True, True, True]]
    assert ar_mask.tolist() == [True, False, False, True, False, False, False]


def test_zero_length_history_layout_reduces_to_the_action_segment():
    input_mask, ar_mask = history_action_masks(
        jnp.zeros((2, 0), dtype=jnp.bool_), action_horizon=3
    )

    assert input_mask.tolist() == [[True, True, True], [True, True, True]]
    assert ar_mask.tolist() == [True, False, False]


def test_history_segment_cannot_attend_to_future_actions_but_actions_can_see_history():
    input_mask, ar_mask = jax.jit(
        lambda mask: history_action_masks(mask, action_horizon=2)
    )(jnp.ones((1, 2), dtype=jnp.bool_))

    attention = make_attn_mask(input_mask, ar_mask)

    assert not bool(attention[0, 0, 2])
    assert bool(attention[0, 2, 0])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"action_horizon": 4, "overlap_O": 4}, "overlap_O"),
        ({"action_horizon": 4, "overlap_O": -1}, "overlap_O"),
        ({"action_horizon": 4, "overlap_O": 1, "history_length": 4}, "history_length"),
        ({"history_length": -1}, "history_length"),
        ({"history_noise_std": -0.1}, "history_noise_std"),
        ({"history_dropout_probability": 1.1}, "history_dropout_probability"),
        ({"history_schedule_warmup_steps": -1}, "history_schedule_warmup_steps"),
        ({"history_schedule_ramp_steps": 0}, "history_schedule_ramp_steps"),
        ({"history_schedule_max_alpha": -0.1}, "history_schedule_max_alpha"),
        ({"continuity_first_order_weight": -1.0}, "continuity_first_order_weight"),
        ({"continuity_second_order_weight": -1.0}, "continuity_second_order_weight"),
        ({"boundary_weight": -1.0}, "boundary_weight"),
        ({"boundary_weight": 0.1, "overlap_O": 0}, "overlap_O"),
        ({"entropy_lambda": 0.1}, "entropy"),
    ],
)
def test_history_configuration_is_validated(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _pi0_config.Pi0Config(**kwargs)


def test_chunkflow_supervision_properties():
    disabled = _pi0_config.Pi0Config(action_horizon=8, overlap_O=2)
    history_enabled = _pi0_config.Pi0Config(action_horizon=8, overlap_O=2, history_length=3)
    boundary_enabled = _pi0_config.Pi0Config(action_horizon=8, overlap_O=2, boundary_weight=0.1)

    assert disabled.chunk_stride == 6
    assert not disabled.chunkflow_supervised_enabled
    assert history_enabled.chunkflow_supervised_enabled
    assert boundary_enabled.chunkflow_supervised_enabled


def test_inputs_spec_includes_history_only_when_configured():
    disabled_obs, _ = _pi0_config.Pi0Config().inputs_spec(batch_size=2)
    enabled_obs, _ = _pi0_config.Pi0Config(
        action_dim=7,
        action_horizon=8,
        overlap_O=2,
        history_length=3,
    ).inputs_spec(batch_size=2)

    assert disabled_obs.action_history is None
    assert disabled_obs.action_history_mask is None
    assert enabled_obs.action_history.shape == (2, 3, 7)
    assert enabled_obs.action_history.dtype == jnp.float32
    assert enabled_obs.action_history_mask.shape == (2, 3)
    assert enabled_obs.action_history_mask.dtype == jnp.bool_


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
