"""Training-only Q/V critics for ChunkFlow AWAC."""

import numbers

from flax import nnx
import jax
import jax.numpy as jnp


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class CriticHead(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_width: int,
        hidden_depth: int,
        *,
        rngs: nnx.Rngs,
    ):
        # Flax 0.10.2 uses plain lists for NNX module sequences.
        self.hidden = [
            nnx.Linear(
                input_dim if index == 0 else hidden_width,
                hidden_width,
                rngs=rngs,
            )
            for index in range(hidden_depth)
        ]
        self.output = nnx.Linear(hidden_width, 1, rngs=rngs)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        hidden = inputs
        for layer in self.hidden:
            hidden = jax.nn.swish(layer(hidden))
        return self.output(hidden)[..., 0]


class ChunkFlowCritic(nnx.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_width: int = 512,
        hidden_depth: int = 2,
        *,
        rngs: nnx.Rngs,
    ):
        self.state_dim = _positive_integer(state_dim, name="state_dim")
        self.action_dim = _positive_integer(action_dim, name="action_dim")
        hidden_width = _positive_integer(hidden_width, name="hidden_width")
        hidden_depth = _positive_integer(hidden_depth, name="hidden_depth")

        self.q = CriticHead(
            self.state_dim + self.action_dim,
            hidden_width,
            hidden_depth,
            rngs=rngs,
        )
        self.v = CriticHead(
            self.state_dim,
            hidden_width,
            hidden_depth,
            rngs=rngs,
        )

    def _validate_state(self, state: jax.Array) -> None:
        if state.ndim != 2 or state.shape[1] != self.state_dim:
            raise ValueError(f"state must have shape [B, {self.state_dim}]")

    def q_value(self, state: jax.Array, action: jax.Array) -> jax.Array:
        self._validate_state(state)
        if action.ndim != 2 or action.shape[1] != self.action_dim:
            raise ValueError(f"action must have shape [B, {self.action_dim}]")
        if action.shape[0] != state.shape[0]:
            raise ValueError("state and action batch dimensions must match")
        return self.q(jnp.concatenate((state, action), axis=-1))

    def value(self, state: jax.Array) -> jax.Array:
        self._validate_state(state)
        return self.v(state)
