"""Dual-objective optimization for step-wise ChunkFlow AWAC."""

import dataclasses

from flax import nnx
from flax import struct
import jax
import jax.numpy as jnp
import optax

from openpi.models import chunkflow_awac
from openpi.models import pi0_config
from openpi.training import chunkflow_batch
from openpi.training import chunkflow_train
from openpi.training import utils as training_utils


@struct.dataclass
class AwacObjectiveOutput:
    reported_total: jax.Array
    actor_objective: jax.Array
    critic_objective: jax.Array
    metrics: dict[str, jax.Array]


def _validate_observation_batch(observation, *, batch_size: int, name: str) -> None:
    state = getattr(observation, "state", None)
    if state is None or not hasattr(state, "shape"):
        raise ValueError(f"{name}.state must have nonempty shape [B, S]")
    if state.ndim != 2 or state.shape[0] != batch_size or state.shape[1] <= 0:
        raise ValueError(
            f"{name}.state must have nonempty shape [{batch_size}, S], got {state.shape}"
        )


def _validate_awac_batch(
    batch: chunkflow_batch.ChunkFlowTrainBatch,
    *,
    action_dim: int,
    action_horizon: int,
) -> None:
    if not isinstance(batch, chunkflow_batch.ChunkFlowTrainBatch):
        raise ValueError("ChunkFlow AWAC requires a composite supervised/transition batch")
    if not isinstance(batch.supervised, chunkflow_batch.PairedChunkBatch):
        raise ValueError("ChunkFlow AWAC composite batch requires a PairedChunkBatch supervised subtree")
    if not isinstance(batch.transition, chunkflow_batch.StepTransitionBatch):
        raise ValueError("ChunkFlow AWAC composite batch requires a StepTransitionBatch transition subtree")

    supervised = batch.supervised
    actions = supervised.actions
    previous_actions = supervised.previous_actions
    if not hasattr(actions, "shape") or not hasattr(previous_actions, "shape"):
        raise ValueError("supervised actions must be arrays with shape [B, action_horizon, action_dim]")
    if (
        actions.ndim != 3
        or actions.shape[0] <= 0
        or actions.shape[1:] != (action_horizon, action_dim)
        or previous_actions.shape != actions.shape
    ):
        raise ValueError(
            "supervised previous/current actions must share nonempty shape "
            f"[B, {action_horizon}, {action_dim}]"
        )
    supervised_batch_size = actions.shape[0]
    _validate_observation_batch(
        supervised.previous_observation,
        batch_size=supervised_batch_size,
        name="supervised.previous_observation",
    )
    _validate_observation_batch(
        supervised.observation,
        batch_size=supervised_batch_size,
        name="supervised.observation",
    )

    transition = batch.transition
    executed_action = transition.executed_action
    if not hasattr(executed_action, "shape") or executed_action.ndim != 2:
        raise ValueError(f"transition.executed_action must have shape [B, {action_dim}]")
    batch_size = executed_action.shape[0]
    if batch_size <= 0 or executed_action.shape != (batch_size, action_dim):
        raise ValueError(f"transition.executed_action must have nonempty shape [B, {action_dim}]")
    for field_name in ("reward", "continuation", "episode_id", "frame_index"):
        value = getattr(transition, field_name)
        if not hasattr(value, "shape") or value.shape != (batch_size,):
            raise ValueError(f"transition.{field_name} must have shape [{batch_size}]")
    _validate_observation_batch(
        transition.observation,
        batch_size=batch_size,
        name="transition.observation",
    )
    _validate_observation_batch(
        transition.next_observation,
        batch_size=batch_size,
        name="transition.next_observation",
    )


def compute_awac_objectives(
    model,
    critic,
    target_value,
    reference_model,
    ema_model,
    batch: chunkflow_batch.ChunkFlowTrainBatch,
    rng: jax.Array,
    model_config: pi0_config.Pi0Config,
) -> AwacObjectiveOutput:
    """Compute disjoint actor and critic objectives from one composite batch."""

    _validate_awac_batch(
        batch,
        action_dim=model.action_dim,
        action_horizon=model.action_horizon,
    )
    (
        supervised_rng,
        state_rng,
        next_state_rng,
        flow_rng,
        noise_rng,
        time_rng,
    ) = jax.random.split(rng, 6)
    supervised, supervised_metrics = model.compute_paired_loss(
        supervised_rng,
        batch.supervised.previous_observation,
        batch.supervised.previous_actions,
        batch.supervised.observation,
        batch.supervised.actions,
        ema_model=ema_model,
        step=batch.supervised.step,
        train=True,
    )

    state = jax.lax.stop_gradient(
        model.encode_critic_state(state_rng, batch.transition.observation, train=True)
    )
    next_state = jax.lax.stop_gradient(
        model.encode_critic_state(next_state_rng, batch.transition.next_observation, train=True)
    )
    q_value = critic.q_value(state, batch.transition.executed_action)
    value = critic.value(state)
    next_value = target_value(next_state)
    target = chunkflow_awac.td_targets(
        batch.transition.reward,
        batch.transition.continuation,
        next_value,
        gamma=model_config.awac_gamma,
    )
    q_loss = jnp.mean(jnp.square(q_value - jax.lax.stop_gradient(target)))
    v_loss, _ = chunkflow_awac.expectile_value_loss(
        q_value,
        value,
        expectile=model_config.awac_expectile_tau_e,
    )
    weight, advantage = chunkflow_awac.clipped_advantage_weights(
        q_value,
        value,
        temperature=model_config.awac_temperature_tau,
        wmax=model_config.awac_wmax,
    )

    executed = batch.transition.executed_action
    batch_size = executed.shape[0]
    full_noise = jax.random.normal(
        noise_rng,
        (batch_size, model.action_horizon, model.action_dim),
        dtype=executed.dtype,
    )
    time = jax.random.beta(time_rng, 1.5, 1.0, (batch_size,)) * 0.999 + 0.001
    dummy_tail = jnp.zeros(
        (batch_size, model.action_horizon - 1, model.action_dim),
        dtype=executed.dtype,
    )
    actor_flow = model.first_action_flow_forward(
        flow_rng,
        batch.transition.observation,
        executed,
        train=True,
        dummy_tail=dummy_tail,
        noise=full_noise,
        time=time,
    )
    reference_flow = reference_model.first_action_flow_forward(
        flow_rng,
        batch.transition.observation,
        executed,
        train=True,
        dummy_tail=dummy_tail,
        noise=full_noise,
        time=time,
    )
    awac_actor = jnp.mean(weight * actor_flow.loss)
    reference = chunkflow_awac.reference_consistency_loss(
        actor_flow.velocity,
        reference_flow.velocity,
    )
    actor_objective = (
        supervised
        + model_config.awac_actor_weight * awac_actor
        + model_config.kl_beta * reference
    )
    critic_objective = model_config.awac_q_weight * q_loss + model_config.awac_v_weight * v_loss
    reported_total = actor_objective + critic_objective
    metrics = {
        "loss": reported_total,
        "actor_objective": actor_objective,
        "critic_objective": critic_objective,
        "awac/actor_loss": awac_actor,
        "awac/q_loss": q_loss,
        "awac/v_loss": v_loss,
        "awac/reference_consistency": reference,
        "awac/q_mean": jnp.mean(q_value),
        "awac/v_mean": jnp.mean(value),
        "awac/td_target_mean": jnp.mean(target),
        "awac/advantage_mean": jnp.mean(advantage),
        "awac/reward_mean": jnp.mean(batch.transition.reward),
        "awac/positive_advantage_fraction": jnp.mean((advantage > 0).astype(jnp.float32)),
        "awac/weight_mean": jnp.mean(weight),
        "awac/weight_max": jnp.max(weight),
        "awac/terminal_fraction": jnp.mean(
            (batch.transition.continuation == 0).astype(jnp.float32)
        ),
        **{f"supervised/{name}": value for name, value in supervised_metrics.items()},
    }
    return AwacObjectiveOutput(
        reported_total=reported_total,
        actor_objective=actor_objective,
        critic_objective=critic_objective,
        metrics=metrics,
    )


def _validate_awac_state(state: training_utils.TrainState, *, history_length: int) -> None:
    for field_name in (
        "critic_params",
        "critic_model_def",
        "critic_opt_state",
        "target_v_params",
        "target_v_model_def",
        "reference_params",
    ):
        if getattr(state, field_name) is None:
            raise ValueError(f"ChunkFlow AWAC training state is missing {field_name}")
    if history_length > 0 and state.ema_params is None:
        raise ValueError("ChunkFlow AWAC training state is missing ema_params")
    if state.ema_decay is not None and state.ema_params is None:
        raise ValueError("ChunkFlow AWAC training state is missing ema_params")


def train_step(config, rng, state, batch):
    """Update actor and critic independently, then advance actor and value EMAs."""

    if not bool(getattr(config.model, "awac_enable", False)):
        raise ValueError("ChunkFlow AWAC train step requires awac_enable=True")
    _validate_awac_state(state, history_length=config.model.history_length)
    _validate_awac_batch(
        batch,
        action_dim=config.model.action_dim,
        action_horizon=config.model.action_horizon,
    )

    model = nnx.merge(state.model_def, state.params)
    critic = nnx.merge(state.critic_model_def, state.critic_params)
    target_value = nnx.merge(state.target_v_model_def, state.target_v_params)
    reference_model = nnx.merge(state.model_def, state.reference_params)
    model.train()
    critic.train()
    target_value.eval()
    reference_model.train()

    ema_model = None
    if state.ema_params is not None:
        ema_model = nnx.merge(state.model_def, state.ema_params)
        ema_model.eval()

    train_rng = jax.random.fold_in(rng, state.step)
    batch = chunkflow_train.batch_with_step(batch, state.step)

    def awac_loss_fn(actor, online_critic):
        output = compute_awac_objectives(
            actor,
            online_critic,
            target_value,
            reference_model,
            ema_model,
            batch,
            train_rng,
            config.model,
        )
        return output.reported_total, output

    (reported_total, objective_output), (actor_grads, critic_grads) = nnx.value_and_grad(
        awac_loss_fn,
        argnums=(
            nnx.DiffState(0, config.trainable_filter),
            nnx.DiffState(1, nnx.Param),
        ),
        has_aux=True,
    )(model, critic)

    actor_params = state.params.filter(config.trainable_filter)
    actor_updates, actor_opt_state = state.tx.update(actor_grads, state.opt_state, actor_params)
    critic_params = nnx.state(critic, nnx.Param)
    critic_updates, critic_opt_state = state.tx.update(
        critic_grads,
        state.critic_opt_state,
        critic_params,
    )
    nnx.update(model, optax.apply_updates(actor_params, actor_updates))
    nnx.update(critic, optax.apply_updates(critic_params, critic_updates))

    new_params = nnx.state(model)
    new_critic_params = nnx.state(critic)
    new_ema_params = state.ema_params
    if state.ema_decay is not None:
        new_ema_params = jax.tree.map(
            lambda old, current: state.ema_decay * old + (1.0 - state.ema_decay) * current,
            state.ema_params,
            new_params,
        )
    new_target_v_params = jax.tree.map(
        lambda old, online: config.model.awac_target_decay * old
        + (1.0 - config.model.awac_target_decay) * online,
        state.target_v_params,
        nnx.state(critic.v),
    )
    new_state = dataclasses.replace(
        state,
        step=state.step + 1,
        params=new_params,
        opt_state=actor_opt_state,
        ema_params=new_ema_params,
        critic_params=new_critic_params,
        critic_opt_state=critic_opt_state,
        target_v_params=new_target_v_params,
        reference_params=state.reference_params,
    )
    metrics = {
        **objective_output.metrics,
        "loss": reported_total,
        "actor_grad_norm": optax.global_norm(actor_grads),
        "critic_grad_norm": optax.global_norm(critic_grads),
    }
    return new_state, metrics
