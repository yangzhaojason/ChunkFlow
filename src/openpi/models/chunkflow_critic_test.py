from flax import nnx
import jax
import jax.numpy as jnp
import pytest

from openpi.models.chunkflow_critic import ChunkFlowCritic


def test_critic_outputs_one_q_and_v_per_transition_under_jit():
    critic = ChunkFlowCritic(
        state_dim=6,
        action_dim=2,
        hidden_width=8,
        hidden_depth=2,
        rngs=nnx.Rngs(0),
    )
    state = jnp.ones((3, 6))
    action = jnp.arange(6, dtype=jnp.float32).reshape(3, 2)
    q, v = jax.jit(lambda s, a: (critic.q_value(s, a), critic.value(s)))(state, action)
    assert q.shape == (3,)
    assert v.shape == (3,)


def test_q_depends_on_action_while_v_does_not_accept_action():
    critic = ChunkFlowCritic(
        state_dim=4,
        action_dim=1,
        hidden_width=8,
        hidden_depth=2,
        rngs=nnx.Rngs(1),
    )
    state = jnp.ones((1, 4))
    q_zero = critic.q_value(state, jnp.zeros((1, 1)))
    q_one = critic.q_value(state, jnp.ones((1, 1)))
    assert q_zero.shape == (1,)
    assert q_one.shape == (1,)
    assert not jnp.allclose(q_zero, q_one)
    assert critic.value(state).shape == (1,)


@pytest.mark.parametrize(
    ("state", "action", "match"),
    [
        (jnp.ones((4,)), jnp.ones((1, 1)), "state"),
        (jnp.ones((1, 5)), jnp.ones((1, 1)), "state"),
        (jnp.ones((1, 4)), jnp.ones((1,)), "action"),
        (jnp.ones((1, 4)), jnp.ones((1, 2)), "action"),
        (jnp.ones((1, 4)), jnp.ones((2, 1)), "batch"),
    ],
)
def test_q_value_rejects_invalid_transition_shapes(state, action, match):
    critic = ChunkFlowCritic(state_dim=4, action_dim=1, rngs=nnx.Rngs(2))

    with pytest.raises(ValueError, match=match):
        critic.q_value(state, action)


@pytest.mark.parametrize("state", [jnp.ones((4,)), jnp.ones((1, 5))])
def test_value_rejects_invalid_state_shapes(state):
    critic = ChunkFlowCritic(state_dim=4, action_dim=1, rngs=nnx.Rngs(3))

    with pytest.raises(ValueError, match="state"):
        critic.value(state)


@pytest.mark.parametrize("field", ["state_dim", "action_dim", "hidden_width", "hidden_depth"])
@pytest.mark.parametrize("value", [True, 0, -1, 1.5])
def test_constructor_rejects_invalid_positive_integer_dimensions(field, value):
    kwargs = {
        "state_dim": 4,
        "action_dim": 2,
        "hidden_width": 8,
        "hidden_depth": 2,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=field):
        ChunkFlowCritic(**kwargs, rngs=nnx.Rngs(4))
