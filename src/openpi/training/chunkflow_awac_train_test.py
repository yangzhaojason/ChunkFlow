import dataclasses
import types
from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest

from openpi.models import model as _model
from openpi.training import chunkflow_batch
from openpi.training.chunkflow_awac_train import compute_awac_objectives


class _FlowOutput(NamedTuple):
    loss: jax.Array
    velocity: jax.Array
    x_t: jax.Array
    time: jax.Array


@dataclasses.dataclass(frozen=True)
class _AwacConfig:
    action_dim: int = 2
    action_horizon: int = 3
    history_length: int = 0
    awac_gamma: float = 0.9
    awac_expectile_tau_e: float = 0.7
    awac_temperature_tau: float = 0.5
    awac_wmax: float = 10.0
    awac_actor_weight: float = 1.25
    awac_q_weight: float = 0.75
    awac_v_weight: float = 1.5
    kl_beta: float = 0.2


class _TinyAwacActor:
    action_dim = 2
    action_horizon = 3

    def __init__(self, scale, calls=None):
        self.scale = jnp.asarray(scale)
        self.calls = calls

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
        del previous_observation, previous_actions, observation, train
        if self.calls is not None:
            self.calls["supervised_rng"] = rng
        ema_anchor = jax.lax.stop_gradient(ema_model.scale)
        loss = jnp.mean(jnp.square(self.scale * actions - 0.25))
        loss = loss + 0.0 * ema_anchor + 0.0 * step
        return loss, {"loss/flow": loss, "history/alpha": jnp.asarray(0.0)}

    def encode_critic_state(self, rng, observation, *, train):
        del train
        if self.calls is not None:
            self.calls.setdefault("state_rngs", []).append(rng)
        state = self.scale * observation.state
        feature = jnp.concatenate((state, self.scale * jnp.ones((state.shape[0], 1))), axis=-1)
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
        del observation, train
        if self.calls is not None:
            self.calls["flow"] = {
                "rng": rng,
                "preprocess_rng": jax.random.split(rng, 3)[0],
                "dummy_tail": dummy_tail,
                "noise": noise,
                "time": time,
            }
        velocity = self.scale * (executed_action + 0.5)
        target_velocity = noise[:, 0] - executed_action
        loss = jnp.mean(jnp.square(velocity - target_velocity), axis=-1)
        x_t = time[:, None] * noise[:, 0] + (1.0 - time[:, None]) * executed_action
        return _FlowOutput(loss=loss, velocity=velocity, x_t=x_t, time=time)


class _HistoryNormalizingActor(_TinyAwacActor):
    def __init__(self, scale):
        super().__init__(scale)
        self.observations_checked = 0

    def _check_observation(self, observation):
        assert observation.action_history is None
        assert observation.action_history_mask is None
        self.observations_checked += 1

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
        self._check_observation(previous_observation)
        self._check_observation(observation)
        return super().compute_paired_loss(
            rng,
            previous_observation,
            previous_actions,
            observation,
            actions,
            ema_model=ema_model,
            step=step,
            train=train,
        )

    def encode_critic_state(self, rng, observation, *, train):
        self._check_observation(observation)
        return super().encode_critic_state(rng, observation, train=train)

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
        self._check_observation(observation)
        return super().first_action_flow_forward(
            rng,
            observation,
            executed_action,
            train=train,
            dummy_tail=dummy_tail,
            noise=noise,
            time=time,
        )


class _TinyCritic:
    def __init__(self, q_scale, v_scale):
        self.q_scale = jnp.asarray(q_scale)
        self.v_scale = jnp.asarray(v_scale)

    def q_value(self, state, action):
        return self.q_scale * (jnp.sum(state, axis=-1) + jnp.sum(action, axis=-1) + 0.5)

    def value(self, state):
        return self.v_scale * (jnp.sum(state, axis=-1) - 0.25)


class _TinyValue:
    def __init__(self, scale):
        self.scale = jnp.asarray(scale)

    def __call__(self, state):
        return self.scale * jnp.sum(state, axis=-1)


class _NeverCalledCritic:
    def __init__(self):
        self.called = False

    def q_value(self, state, action):
        del state, action
        self.called = True
        raise AssertionError("critic must not run for an invalid transition batch")

    def value(self, state):
        del state
        self.called = True
        raise AssertionError("critic must not run for an invalid transition batch")


class _NeverCalledModel:
    action_dim = 2
    action_horizon = 3

    def __init__(self):
        self.called = False

    def _fail(self, *args, **kwargs):
        del args, kwargs
        self.called = True
        raise AssertionError("model must not run for an invalid transition batch")

    compute_paired_loss = _fail
    encode_critic_state = _fail
    first_action_flow_forward = _fail


class _ReferenceMustNotRun:
    def __init__(self):
        self.called = False

    def first_action_flow_forward(self, *args, **kwargs):
        del args, kwargs
        self.called = True
        raise AssertionError("zero reference coefficient must skip the reference forward")


def _observation(state):
    state = jnp.asarray(state, dtype=jnp.float32)
    return _model.Observation(images={}, image_masks={}, state=state)


def _tiny_composite_batch():
    previous_observation = _observation([[0.0, 0.1], [0.2, -0.1]])
    observation = _observation([[0.2, -0.1], [0.4, 0.3]])
    next_observation = _observation([[0.3, 0.0], [0.5, 0.2]])
    return chunkflow_batch.ChunkFlowTrainBatch(
        supervised=chunkflow_batch.PairedChunkBatch(
            previous_observation=previous_observation,
            previous_actions=jnp.full((2, 3, 2), 0.1, dtype=jnp.float32),
            observation=observation,
            actions=jnp.full((2, 3, 2), 0.3, dtype=jnp.float32),
            step=jnp.asarray(0, dtype=jnp.int32),
        ),
        transition=chunkflow_batch.StepTransitionBatch(
            observation=observation,
            executed_action=jnp.asarray([[0.2, 0.4], [-0.3, 0.1]], dtype=jnp.float32),
            reward=jnp.asarray([0.5, -0.2], dtype=jnp.float32),
            continuation=jnp.asarray([1.0, 0.0], dtype=jnp.float32),
            next_observation=next_observation,
            episode_id=jnp.asarray([4, 4], dtype=jnp.int32),
            frame_index=jnp.asarray([7, 8], dtype=jnp.int32),
        ),
    )


def _batch_with_history(batch, history_length):
    def add_history(observation):
        batch_size, action_dim = observation.state.shape
        return observation.replace(
            action_history=jnp.zeros(
                (batch_size, history_length, action_dim), dtype=jnp.float32
            ),
            action_history_mask=jnp.zeros(
                (batch_size, history_length), dtype=jnp.bool_
            ),
        )

    return batch.replace(
        supervised=batch.supervised.replace(
            previous_observation=add_history(batch.supervised.previous_observation),
            observation=add_history(batch.supervised.observation),
        ),
        transition=batch.transition.replace(
            observation=add_history(batch.transition.observation),
            next_observation=add_history(batch.transition.next_observation),
        ),
    )


def _observation_at(batch, role):
    if role == "supervised.previous_observation":
        return batch.supervised.previous_observation
    if role == "supervised.observation":
        return batch.supervised.observation
    if role == "transition.observation":
        return batch.transition.observation
    if role == "transition.next_observation":
        return batch.transition.next_observation
    raise AssertionError(f"unknown observation role: {role}")


def _replace_observation_at(batch, role, observation):
    if role == "supervised.previous_observation":
        return batch.replace(
            supervised=batch.supervised.replace(previous_observation=observation)
        )
    if role == "supervised.observation":
        return batch.replace(supervised=batch.supervised.replace(observation=observation))
    if role == "transition.observation":
        return batch.replace(transition=batch.transition.replace(observation=observation))
    if role == "transition.next_observation":
        return batch.replace(
            transition=batch.transition.replace(next_observation=observation)
        )
    raise AssertionError(f"unknown observation role: {role}")


def test_awac_objectives_keep_actor_and_critic_gradients_disjoint():
    batch = _tiny_composite_batch()

    def loss(actor_scale, q_scale, v_scale, reference_scale):
        actor = _TinyAwacActor(actor_scale)
        critic = _TinyCritic(q_scale, v_scale)
        result = compute_awac_objectives(
            actor,
            critic,
            _TinyValue(0.25),
            _TinyAwacActor(reference_scale),
            _TinyAwacActor(1.0),
            batch,
            jax.random.key(0),
            _AwacConfig(),
        )
        return result.actor_objective, result.critic_objective

    actor_from_actor, q_from_actor, v_from_actor, ref_from_actor = jax.grad(
        lambda *args: loss(*args)[0], argnums=(0, 1, 2, 3)
    )(1.0, 2.0, 0.5, 0.75)
    actor_from_critic, q_from_critic, v_from_critic = jax.grad(
        lambda a, q, v: loss(a, q, v, 0.75)[1], argnums=(0, 1, 2)
    )(1.0, 2.0, 0.5)

    assert actor_from_actor != 0
    assert q_from_actor == 0
    assert v_from_actor == 0
    assert ref_from_actor == 0
    assert actor_from_critic == 0
    assert q_from_critic != 0
    assert v_from_critic != 0


def test_awac_objectives_share_exact_flow_random_inputs():
    batch = _tiny_composite_batch()
    current_calls = {}
    reference_calls = {}
    rng = jax.random.key(17)
    result = compute_awac_objectives(
        _TinyAwacActor(1.0, current_calls),
        _TinyCritic(2.0, 0.5),
        _TinyValue(0.25),
        _TinyAwacActor(0.75, reference_calls),
        _TinyAwacActor(1.0),
        batch,
        rng,
        _AwacConfig(),
    )

    supervised_rng, state_rng, next_state_rng, flow_rng, noise_rng, time_rng = jax.random.split(rng, 6)
    expected_noise = jax.random.normal(noise_rng, (2, 3, 2), dtype=jnp.float32)
    expected_time = jax.random.beta(time_rng, 1.5, 1.0, (2,)) * 0.999 + 0.001
    expected_preprocess_rng = jax.random.split(flow_rng, 3)[0]
    current_flow = current_calls["flow"]
    reference_flow = reference_calls["flow"]

    assert jnp.array_equal(current_calls["supervised_rng"], supervised_rng)
    assert jnp.array_equal(current_calls["state_rngs"][0], state_rng)
    assert jnp.array_equal(current_calls["state_rngs"][1], next_state_rng)
    assert jnp.array_equal(current_flow["rng"], flow_rng)
    assert jnp.array_equal(reference_flow["rng"], flow_rng)
    assert jnp.array_equal(current_flow["preprocess_rng"], expected_preprocess_rng)
    assert jnp.array_equal(reference_flow["preprocess_rng"], expected_preprocess_rng)
    assert current_flow["noise"] is reference_flow["noise"]
    assert current_flow["time"] is reference_flow["time"]
    assert current_flow["dummy_tail"] is reference_flow["dummy_tail"]
    assert jnp.array_equal(current_flow["noise"], expected_noise)
    assert jnp.array_equal(current_flow["time"], expected_time)
    assert jnp.array_equal(current_flow["dummy_tail"], jnp.zeros((2, 2, 2), dtype=jnp.float32))
    assert all(jnp.isfinite(value) for value in result.metrics.values())


def test_awac_objectives_skip_reference_forward_when_coefficient_is_zero():
    reference = _ReferenceMustNotRun()
    result = compute_awac_objectives(
        _TinyAwacActor(1.0),
        _TinyCritic(2.0, 0.5),
        _TinyValue(0.25),
        reference,
        _TinyAwacActor(1.0),
        _tiny_composite_batch(),
        jax.random.key(23),
        dataclasses.replace(_AwacConfig(), kl_beta=0.0),
    )

    reference_metric = result.metrics["awac/reference_consistency"]
    assert not reference.called
    assert reference_metric.shape == ()
    assert reference_metric.dtype == jnp.float32
    assert reference_metric == 0


def test_awac_objectives_normalize_zero_length_histories_before_model_calls():
    batch = _batch_with_history(_tiny_composite_batch(), history_length=0)
    actor = _HistoryNormalizingActor(1.0)
    reference = _HistoryNormalizingActor(0.75)

    result = compute_awac_objectives(
        actor,
        _TinyCritic(2.0, 0.5),
        _TinyValue(0.25),
        reference,
        _TinyAwacActor(1.0),
        batch,
        jax.random.key(37),
        _AwacConfig(history_length=0),
    )

    assert jnp.isfinite(result.reported_total)
    assert actor.observations_checked == 5
    assert reference.observations_checked == 1
    for role in (
        "supervised.previous_observation",
        "supervised.observation",
        "transition.observation",
        "transition.next_observation",
    ):
        observation = _observation_at(batch, role)
        assert observation.action_history.shape == (2, 0, 2)
        assert observation.action_history_mask.shape == (2, 0)


@pytest.mark.parametrize(
    "role",
    [
        "supervised.previous_observation",
        "supervised.observation",
        "transition.observation",
        "transition.next_observation",
    ],
)
@pytest.mark.parametrize("violation", ["missing", "wrong_length"])
def test_awac_objectives_require_configured_history_on_every_observation(role, violation):
    batch = _batch_with_history(_tiny_composite_batch(), history_length=2)
    observation = _observation_at(batch, role)
    if violation == "missing":
        observation = observation.replace(
            action_history=None,
            action_history_mask=None,
        )
    else:
        observation = observation.replace(
            action_history=jnp.zeros((2, 1, 2), dtype=jnp.float32),
            action_history_mask=jnp.zeros((2, 1), dtype=jnp.bool_),
        )
    batch = _replace_observation_at(batch, role, observation)
    model = _NeverCalledModel()
    critic = _NeverCalledCritic()

    with pytest.raises(ValueError, match=rf"{role}.*action_history"):
        compute_awac_objectives(
            model,
            critic,
            _TinyValue(0.25),
            _TinyAwacActor(0.75),
            _TinyAwacActor(1.0),
            batch,
            jax.random.key(41),
            _AwacConfig(history_length=2),
        )

    assert not model.called
    assert not critic.called


@pytest.mark.parametrize(
    "role",
    [
        "supervised.previous_observation",
        "supervised.observation",
        "transition.observation",
        "transition.next_observation",
    ],
)
def test_awac_objectives_reject_nonzero_history_when_disabled(role):
    batch = _tiny_composite_batch()
    observation = _observation_at(batch, role).replace(
        action_history=jnp.zeros((2, 1, 2), dtype=jnp.float32),
        action_history_mask=jnp.zeros((2, 1), dtype=jnp.bool_),
    )
    batch = _replace_observation_at(batch, role, observation)
    model = _NeverCalledModel()
    critic = _NeverCalledCritic()

    with pytest.raises(ValueError, match=rf"{role}.*action_history"):
        compute_awac_objectives(
            model,
            critic,
            _TinyValue(0.25),
            _TinyAwacActor(0.75),
            _TinyAwacActor(1.0),
            batch,
            jax.random.key(43),
            _AwacConfig(history_length=0),
        )

    assert not model.called
    assert not critic.called


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("empty_action", "executed_action"),
        ("reward_rank", "reward"),
        ("continuation_batch", "continuation"),
        ("episode_rank", "episode_id"),
        ("frame_batch", "frame_index"),
        ("observation_batch", "transition.observation.state"),
        ("next_observation_batch", "transition.next_observation.state"),
    ],
)
def test_awac_objectives_validate_transition_shapes_before_critic(case, message):
    batch = _tiny_composite_batch()
    transition = batch.transition
    if case == "empty_action":
        transition = transition.replace(executed_action=jnp.zeros((0, 2), dtype=jnp.float32))
    elif case == "reward_rank":
        transition = transition.replace(reward=jnp.zeros((2, 1), dtype=jnp.float32))
    elif case == "continuation_batch":
        transition = transition.replace(continuation=jnp.zeros((1,), dtype=jnp.float32))
    elif case == "episode_rank":
        transition = transition.replace(episode_id=jnp.zeros((2, 1), dtype=jnp.int32))
    elif case == "frame_batch":
        transition = transition.replace(frame_index=jnp.zeros((1,), dtype=jnp.int32))
    elif case == "observation_batch":
        transition = transition.replace(observation=_observation([[0.0, 0.0]]))
    elif case == "next_observation_batch":
        transition = transition.replace(next_observation=_observation([[0.0, 0.0]]))
    invalid_batch = batch.replace(transition=transition)
    critic = _NeverCalledCritic()

    with pytest.raises(ValueError, match=message):
        compute_awac_objectives(
            _TinyAwacActor(1.0),
            critic,
            _TinyValue(0.25),
            _TinyAwacActor(0.75),
            _TinyAwacActor(1.0),
            invalid_batch,
            jax.random.key(0),
            _AwacConfig(),
        )

    assert not critic.called


def test_awac_objectives_allow_images_without_explicit_masks():
    batch = _tiny_composite_batch()
    observation = batch.transition.observation.replace(
        images={"camera": jnp.zeros((2, 4, 4, 3), dtype=jnp.float32)},
        image_masks={},
    )

    result = compute_awac_objectives(
        _TinyAwacActor(1.0),
        _TinyCritic(2.0, 0.5),
        _TinyValue(0.25),
        _TinyAwacActor(0.75),
        _TinyAwacActor(1.0),
        batch.replace(transition=batch.transition.replace(observation=observation)),
        jax.random.key(31),
        _AwacConfig(),
    )

    assert jnp.isfinite(result.reported_total)


def _malformed_optional_observation(case):
    observation = _observation([[0.2, -0.1], [0.4, 0.3]])
    values = {
        field.name: getattr(observation, field.name)
        for field in dataclasses.fields(observation)
    }

    def replace(**updates):
        return types.SimpleNamespace(**(values | updates))

    if case == "state_width":
        return replace(state=jnp.zeros((2, 3), dtype=jnp.float32))
    if case == "image_batch":
        return replace(
            images={"camera": jnp.zeros((1, 4, 4, 3), dtype=jnp.float32)},
            image_masks={"camera": jnp.ones((2,), dtype=jnp.bool_)},
        )
    if case == "image_mask_batch":
        return replace(
            images={"camera": jnp.zeros((2, 4, 4, 3), dtype=jnp.float32)},
            image_masks={"camera": jnp.ones((1,), dtype=jnp.bool_)},
        )
    if case == "image_key_pair":
        return replace(
            images={"camera": jnp.zeros((2, 4, 4, 3), dtype=jnp.float32)},
            image_masks={"other": jnp.ones((2,), dtype=jnp.bool_)},
        )
    if case == "token_batch":
        return replace(
            tokenized_prompt=jnp.zeros((1, 4), dtype=jnp.int32),
            tokenized_prompt_mask=jnp.ones((1, 4), dtype=jnp.bool_),
        )
    if case == "prompt_mask_shape":
        return replace(
            tokenized_prompt=jnp.zeros((2, 4), dtype=jnp.int32),
            tokenized_prompt_mask=jnp.ones((2, 3), dtype=jnp.bool_),
        )
    if case == "token_ar_batch":
        return replace(
            tokenized_prompt=jnp.zeros((2, 4), dtype=jnp.int32),
            tokenized_prompt_mask=jnp.ones((2, 4), dtype=jnp.bool_),
            token_ar_mask=jnp.zeros((1, 4), dtype=jnp.int32),
        )
    if case == "token_loss_batch":
        return replace(
            tokenized_prompt=jnp.zeros((2, 4), dtype=jnp.int32),
            tokenized_prompt_mask=jnp.ones((2, 4), dtype=jnp.bool_),
            token_loss_mask=jnp.ones((1, 4), dtype=jnp.bool_),
        )
    if case == "history_batch":
        return replace(
            action_history=jnp.zeros((1, 2, 2), dtype=jnp.float32),
            action_history_mask=jnp.ones((1, 2), dtype=jnp.bool_),
        )
    if case == "history_mask_shape":
        return replace(
            action_history=jnp.zeros((2, 2, 2), dtype=jnp.float32),
            action_history_mask=jnp.ones((2, 1), dtype=jnp.bool_),
        )
    if case == "rewards_batch":
        return replace(rewards=jnp.zeros((1, 3), dtype=jnp.float32))
    if case == "discounts_batch":
        return replace(discounts=jnp.zeros((1, 3), dtype=jnp.float32))
    if case == "executed_actions_batch":
        return replace(executed_actions=jnp.zeros((1, 3, 2), dtype=jnp.float32))
    if case == "prev_tail_batch":
        return replace(prev_chunk_tail_actions=jnp.zeros((1, 1, 2), dtype=jnp.float32))
    raise AssertionError(f"unknown malformed observation case: {case}")


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("state_width", "state"),
        ("image_batch", "camera"),
        ("image_mask_batch", "image_masks"),
        ("image_key_pair", "image"),
        ("token_batch", "tokenized_prompt"),
        ("prompt_mask_shape", "tokenized_prompt_mask"),
        ("token_ar_batch", "token_ar_mask"),
        ("token_loss_batch", "token_loss_mask"),
        ("history_batch", "action_history"),
        ("history_mask_shape", "action_history_mask"),
        ("rewards_batch", "rewards"),
        ("discounts_batch", "discounts"),
        ("executed_actions_batch", "executed_actions"),
        ("prev_tail_batch", "prev_chunk_tail_actions"),
    ],
)
def test_awac_objectives_validate_all_observation_leaves_before_model_forward(case, message):
    batch = _tiny_composite_batch()
    invalid_batch = batch.replace(
        transition=batch.transition.replace(observation=_malformed_optional_observation(case))
    )
    model = _NeverCalledModel()
    critic = _NeverCalledCritic()

    with pytest.raises(ValueError, match=message):
        compute_awac_objectives(
            model,
            critic,
            _TinyValue(0.25),
            _TinyAwacActor(0.75),
            _TinyAwacActor(1.0),
            invalid_batch,
            jax.random.key(0),
            _AwacConfig(),
        )

    assert not model.called
    assert not critic.called
