"""Pi0/Pi05 模型（JAX/Flax NNX）。

结构概览：
- VLM 主干：PaliGemma（通过 bridge 连接到 NNX）
- 动作专家：Gemma expert（与主干混合注意力）
- 图像编码：SigLIP（bridge 连接）
- 两种模式：Pi0（状态连续、无 adaRMS）与 Pi05（状态离散进 token，动作专家用 adaRMS 条件）
"""

from __future__ import annotations

import logging
import numbers
from typing import NamedTuple

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models.chunkflow_history import corrupt_history
from openpi.models.chunkflow_history import scheduled_sampling_alpha
from openpi.models.chunkflow_history import validate_history
from openpi.models.chunkflow_losses import boundary_consistency_loss
from openpi.models.chunkflow_losses import continuity_penalties
from openpi.models.chunkflow_losses import flow_endpoint
from openpi.models.chunkflow_objectives import combine_supervised_losses
from openpi.models.chunkflow_objectives import coordinate_aligned_predicted_history
from openpi.models.chunkflow_objectives import share_boundary_noise
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

# 项目日志记录器
logger = logging.getLogger("openpi")


class FlowForwardOutput(NamedTuple):
    per_step_loss: jax.Array
    velocity: jax.Array
    x_t: jax.Array
    time: jax.Array
    endpoint: jax.Array


class FirstActionFlowOutput(NamedTuple):
    loss: jax.Array
    velocity: jax.Array
    x_t: jax.Array
    time: jax.Array


def _require_exact_boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")
    return value


def _validate_floating_finite(array: jax.Array, *, name: str) -> None:
    if not jnp.issubdtype(array.dtype, jnp.floating):
        raise ValueError(f"{name} must use a real floating dtype")
    if isinstance(array, (jax.core.Tracer, jax.ShapeDtypeStruct)):
        return
    if not np.all(np.isfinite(np.asarray(array))):
        raise ValueError(f"{name} must contain only finite values")


def history_action_masks(
    history_mask: jax.Array,
    *,
    action_horizon: int,
    strict_action_causality: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Build validity and segment-start masks for history followed by future actions."""

    strict_action_causality = _require_exact_boolean(
        strict_action_causality,
        name="strict_action_causality",
    )
    history_mask = jnp.asarray(history_mask)
    if history_mask.ndim != 2 or history_mask.dtype != jnp.bool_:
        raise ValueError("history_mask must be bool [B, P]")
    if (
        isinstance(action_horizon, bool)
        or not isinstance(action_horizon, numbers.Integral)
        or action_horizon <= 0
    ):
        raise ValueError("action_horizon must be a positive integer")

    action_mask = jnp.ones((history_mask.shape[0], action_horizon), dtype=jnp.bool_)
    input_mask = jnp.concatenate([history_mask, action_mask], axis=1)
    history_length = history_mask.shape[1]
    history_ar = (
        jnp.concatenate(
            [jnp.ones((1,), dtype=jnp.bool_), jnp.zeros((history_length - 1,), dtype=jnp.bool_)]
        )
        if history_length > 0
        else jnp.zeros((0,), dtype=jnp.bool_)
    )
    action_ar = (
        jnp.ones((action_horizon,), dtype=jnp.bool_)
        if strict_action_causality
        else jnp.concatenate(
            [
                jnp.ones((1,), dtype=jnp.bool_),
                jnp.zeros((action_horizon - 1,), dtype=jnp.bool_),
            ]
        )
    )
    return input_mask, jnp.concatenate([history_ar, action_ar])


def make_attn_mask(input_mask, mask_ar):
    """从 big_vision 改编的注意力掩码构造：支持 prefix-lm/因果/分块等模式。

    - 输入：
      - input_mask: bool[B, N]，有效 token（True）/padding（False）
      - mask_ar: bool[?B, N]，True 表示“新段开始”，False 表示延续前一段
    - 语义：仅允许关注到“段编号不大于自身”的位置，同时受 input_mask 约束
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """对标量位置生成正余弦位置嵌入（维度须为偶数）。"""
    if embedding_dim % 2 != 0:
        raise ValueError(
            f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        # 父类包含动作维度/时域/最大 token 长度等基础参数
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        # Pi05 开关：影响状态处理与 adaRMS 条件
        self.pi05 = config.pi05
        self.history_length = config.history_length
        self.history_noise_std = config.history_noise_std
        self.history_dropout_probability = config.history_dropout_probability
        self.history_schedule_warmup_steps = config.history_schedule_warmup_steps
        self.history_schedule_ramp_steps = config.history_schedule_ramp_steps
        self.history_schedule_max_alpha = config.history_schedule_max_alpha
        self.overlap_O = config.overlap_O
        self.boundary_weight = config.boundary_weight
        # Continuity regularization weights (ChunkFlow-style, no-RL)
        self.continuity_first_order_weight = getattr(
            config, "continuity_first_order_weight", 0.0
        )
        self.continuity_second_order_weight = getattr(
            config, "continuity_second_order_weight", 0.0
        )
        # 取出 VLM 主干与动作专家的配置（宽度/深度/头数等）
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        self.critic_state_dim = (
            paligemma_config.width + action_expert_config.width + config.action_dim
        )
        # TODO：未来改为原生 NNX；当前通过 bridge 连接 Gemma 模块
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        # 懒初始化，Pi05 时仅为动作专家启用 adaRMS
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[
                      False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        # 以假观测触发图像编码器的 lazy_init
        img.lazy_init(next(iter(config.fake_obs().images.values())),
                      train=False, rngs=rngs)
        # 聚合 VLM 与图像编码器
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        # 动作/时间相关线性层
        self.action_in_proj = nnx.Linear(
            config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(
                action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(
                action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(
                config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(
                2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(
                action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(
            action_expert_config.width, config.action_dim, rngs=rngs)

        # 由 model.train()/eval() 自动设置，用于控制 dropout 等是否确定性
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        """将图像与语言嵌入并拼接为前缀序列，返回嵌入与掩码。"""
        input_mask = []
        ar_mask = []
        tokens = []
        # 图像嵌入
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # 图像 token 之间互相可见
            ar_mask += [False] * image_tokens.shape[1]

        # 添加语言 token（若存在）
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(
                obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # 图像与语言之间完全可见
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    def encode_critic_state(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        train: bool,
    ) -> jax.Array:
        """Encode a stopped actor feature for the training-only ChunkFlow critic."""

        state = observation.state
        if state.ndim != 2 or state.shape[1] != self.action_dim or state.shape[0] <= 0:
            raise ValueError(f"state must have shape [B, {self.action_dim}] with B > 0")
        _validate_floating_finite(state, name="state")
        batch_size = state.shape[0]
        if any(image.shape[0] != batch_size for image in observation.images.values()):
            raise ValueError("state batch dimension must match observation images")

        has_history = observation.action_history is not None
        has_history_mask = observation.action_history_mask is not None
        if has_history != has_history_mask:
            raise ValueError("action_history and action_history_mask must be provided together")
        if self.history_length == 0:
            if has_history:
                raise ValueError("action history fields must be absent when history is disabled")
        else:
            if not has_history:
                raise ValueError("action history and mask are required when history is enabled")
            validate_history(
                observation.action_history,
                observation.action_history_mask,
                action_dim=self.action_dim,
            )
            if observation.action_history.shape[0] != batch_size:
                raise ValueError("action_history batch dimension must match state")
            if observation.action_history.shape[1] != self.history_length:
                raise ValueError(
                    f"action_history length {observation.action_history.shape[1]} does not match "
                    f"configured history_length {self.history_length}"
                )
            _validate_floating_finite(observation.action_history, name="action_history")

        observation = _model.preprocess_observation(rng, observation, train=train)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attention = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_output, _), _ = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attention,
            positions=positions,
        )
        prefix_weight = prefix_mask[..., None].astype(prefix_output.dtype)
        prefix_pool = jnp.sum(prefix_output * prefix_weight, axis=1) / jnp.maximum(
            jnp.sum(prefix_weight, axis=1),
            1.0,
        )

        if self.history_length == 0:
            history_pool = jnp.zeros(
                (batch_size, self.action_in_proj.out_features),
                dtype=prefix_pool.dtype,
            )
        else:
            history_output = self.action_in_proj(observation.action_history)
            history_weight = observation.action_history_mask[..., None].astype(
                history_output.dtype
            )
            history_pool = jnp.sum(history_output * history_weight, axis=1) / jnp.maximum(
                jnp.sum(history_weight, axis=1),
                1.0,
            )

        feature = jnp.concatenate([prefix_pool, history_pool, observation.state], axis=-1)
        feature = feature.astype(jnp.float32)
        expected_shape = (batch_size, self.critic_state_dim)
        if feature.shape != expected_shape:
            raise ValueError(
                f"critic state feature must have shape {expected_shape}, got {feature.shape}"
            )
        return jax.lax.stop_gradient(feature)

    @at.typecheck
    def embed_suffix(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
        *,
        strict_action_causality: bool = False,
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        """将状态/动作/时间编码为后缀序列，返回嵌入、掩码以及 adaRMS 条件（Pi05）。"""
        strict_action_causality = _require_exact_boolean(
            strict_action_causality,
            name="strict_action_causality",
        )
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # Pi0：加入单个连续状态 token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(
                jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # 图像/语言不去关注 state/action
            ar_mask.append(jnp.array([True], dtype=jnp.bool_))

        has_history = obs.action_history is not None
        has_history_mask = obs.action_history_mask is not None
        if has_history != has_history_mask:
            raise ValueError("action_history and action_history_mask must be provided together")
        if has_history:
            validate_history(obs.action_history, obs.action_history_mask, action_dim=self.action_dim)
            if obs.action_history.shape[0] != noisy_actions.shape[0]:
                raise ValueError("action_history batch dimension must match noisy_actions")
            if obs.action_history.shape[1] != self.history_length:
                raise ValueError(
                    f"action_history length {obs.action_history.shape[1]} does not match configured "
                    f"history_length {self.history_length}"
                )
            history_tokens = self.action_in_proj(obs.action_history)
            tokens.append(history_tokens)

        action_tokens = self.action_in_proj(noisy_actions)
        # 时间步正余弦位置编码（敏感度 [0,1]）
        time_emb = posemb_sincos(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # Pi05：时间 MLP 生成 adaRMS 条件
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # Pi0：时间扩展到动作序列，与 action 级联后过 MLP 融合
            time_tokens = einops.repeat(
                time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate(
                [action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        if has_history:
            history_and_action_mask, history_and_action_ar_mask = history_action_masks(
                obs.action_history_mask,
                action_horizon=self.action_horizon,
                strict_action_causality=strict_action_causality,
            )
            input_mask.append(history_and_action_mask)
            ar_mask.append(history_and_action_ar_mask)
        else:
            input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
            # 图像/语言/状态不去关注 action tokens（第一位遮蔽，后续允许因果）
            ar_mask.append(
                jnp.ones((self.action_horizon,), dtype=jnp.bool_)
                if strict_action_causality
                else jnp.concatenate(
                    [
                        jnp.ones((1,), dtype=jnp.bool_),
                        jnp.zeros((self.action_horizon - 1,), dtype=jnp.bool_),
                    ]
                )
            )
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.concatenate(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    def _predict_velocity(
        self,
        observation: _model.Observation,
        x_t: _model.Actions,
        time: jax.Array,
        *,
        strict_action_causality: bool = False,
    ) -> jax.Array:
        strict_action_causality = _require_exact_boolean(
            strict_action_causality,
            name="strict_action_causality",
        )
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation,
            x_t,
            time,
            strict_action_causality=strict_action_causality,
        )
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )
        return self.action_out_proj(suffix_out[:, -self.action_horizon :])

    def flow_forward(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool,
        noise: jax.Array | None = None,
        time: jax.Array | None = None,
        strict_action_causality: bool = False,
    ) -> FlowForwardOutput:
        """Run one flow-matching forward and expose its endpoint estimate."""

        strict_action_causality = _require_exact_boolean(
            strict_action_causality,
            name="strict_action_causality",
        )
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        if actions.ndim != 3 or actions.shape[-2:] != (self.action_horizon, self.action_dim):
            raise ValueError(
                f"actions must have shape [B, {self.action_horizon}, {self.action_dim}], got {actions.shape}"
            )

        batch_shape = actions.shape[:-2]
        if noise is None:
            noise = jax.random.normal(noise_rng, actions.shape, dtype=actions.dtype)
        elif noise.shape != actions.shape:
            raise ValueError(f"noise shape {noise.shape} must match actions {actions.shape}")
        if time is None:
            time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        elif time.shape != batch_shape:
            raise ValueError(f"time shape {time.shape} must match batch shape {batch_shape}")

        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1.0 - time_expanded) * actions
        target_velocity = noise - actions
        velocity = self._predict_velocity(
            observation,
            x_t,
            time,
            strict_action_causality=strict_action_causality,
        )
        per_step_loss = jnp.mean(jnp.square(velocity - target_velocity), axis=-1)
        endpoint = flow_endpoint(x_t, velocity, time)
        return FlowForwardOutput(per_step_loss, velocity, x_t, time, endpoint)

    def first_action_flow_forward(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        executed_action: jax.Array,
        *,
        train: bool,
        dummy_tail: jax.Array | None = None,
        noise: jax.Array | None = None,
        time: jax.Array | None = None,
    ) -> FirstActionFlowOutput:
        """Run strictly causal flow matching for one explicitly executed action."""

        if (
            executed_action.ndim != 2
            or executed_action.shape[-1] != self.action_dim
            or executed_action.shape[0] <= 0
        ):
            raise ValueError(
                f"executed_action must have shape [B, {self.action_dim}] with B > 0"
            )
        _validate_floating_finite(executed_action, name="executed_action")
        batch_size = executed_action.shape[0]

        expected_tail = (batch_size, self.action_horizon - 1, self.action_dim)
        if dummy_tail is None:
            dummy_tail = jnp.zeros(expected_tail, dtype=executed_action.dtype)
        if dummy_tail.shape != expected_tail:
            raise ValueError(f"dummy_tail must have shape {expected_tail}")
        _validate_floating_finite(dummy_tail, name="dummy_tail")
        if dummy_tail.dtype != executed_action.dtype:
            raise ValueError(
                f"dummy_tail dtype {dummy_tail.dtype} must match executed_action dtype "
                f"{executed_action.dtype}"
            )

        actions = jnp.concatenate([executed_action[:, None], dummy_tail], axis=1)
        if noise is not None:
            if noise.shape != actions.shape:
                raise ValueError(f"noise shape {noise.shape} must match actions {actions.shape}")
            _validate_floating_finite(noise, name="noise")
            if noise.dtype != actions.dtype:
                raise ValueError(
                    f"noise dtype {noise.dtype} must match actions dtype {actions.dtype}"
                )
        if time is not None:
            if time.shape != (batch_size,):
                raise ValueError(f"time must have shape ({batch_size},)")
            _validate_floating_finite(time, name="time")

        output = self.flow_forward(
            rng,
            observation,
            actions,
            train=train,
            noise=noise,
            time=time,
            strict_action_causality=True,
        )
        return FirstActionFlowOutput(
            loss=output.per_step_loss[:, 0],
            velocity=output.velocity[:, 0],
            x_t=output.x_t[:, 0],
            time=output.time,
        )

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        output = self.flow_forward(rng, observation, actions, train=train)
        per_pos_mse = output.per_step_loss

        # ChunkFlow continuity regularization (no-RL): extrapolate the sampled
        # flow field to its clean endpoint and regularize that action sequence.
        lam_tv = getattr(self, "continuity_first_order_weight", 0.0)
        lam_dd2 = getattr(self, "continuity_second_order_weight", 0.0)

        if lam_tv > 0.0 or lam_dd2 > 0.0:
            tv_loss, dd2_loss = continuity_penalties(output.endpoint)

            # Aggregate to per-sample scalar and return. Keep base per-position loss by averaging over time.
            base_per_sample = jnp.mean(per_pos_mse, axis=-1)  # [B]
            return base_per_sample + lam_tv * tv_loss + lam_dd2 * dd2_loss

        # Default: return per-position loss (keeps existing behavior)
        return per_pos_mse

    def compute_paired_loss(
        self,
        rng: at.KeyArrayLike,
        previous_observation: _model.Observation,
        previous_actions: _model.Actions,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        ema_model: Pi0 | None,
        step: int | jax.Array,
        train: bool,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Train on adjacent chunks with scheduled history and seam supervision."""

        if previous_actions.shape != actions.shape or actions.ndim != 3:
            raise ValueError("paired actions must share shape [B, L, A]")
        if actions.shape[-2:] != (self.action_horizon, self.action_dim):
            raise ValueError("paired actions do not match the configured horizon/action dimension")

        (
            previous_noise_rng,
            current_noise_rng,
            time_rng,
            previous_forward_rng,
            ema_forward_rng,
            current_forward_rng,
            history_rng,
        ) = jax.random.split(rng, 7)
        previous_noise = jax.random.normal(
            previous_noise_rng, previous_actions.shape, dtype=previous_actions.dtype
        )
        current_noise = jax.random.normal(
            current_noise_rng, actions.shape, dtype=actions.dtype
        )
        previous_noise = share_boundary_noise(
            previous_noise,
            current_noise,
            overlap=self.overlap_O,
        )
        time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001

        previous_output = None
        if self.boundary_weight > 0:
            previous_output = self.flow_forward(
                previous_forward_rng,
                previous_observation,
                previous_actions,
                train=train,
                noise=previous_noise,
                time=time,
            )

        alpha = jnp.zeros((), dtype=actions.dtype)
        if self.history_length > 0:
            if ema_model is None:
                raise ValueError("EMA model is required when history conditioning is enabled")
            if observation.action_history is None or observation.action_history_mask is None:
                raise ValueError("current paired observation requires action_history and mask")
            ema_previous = ema_model.flow_forward(
                ema_forward_rng,
                previous_observation,
                previous_actions,
                train=False,
                noise=previous_noise,
                time=time,
            )
            predicted_history, prediction_mask = coordinate_aligned_predicted_history(
                ema_previous.endpoint,
                previous_actions,
                observation.action_history,
                stride=self.action_horizon - self.overlap_O,
            )
            alpha = scheduled_sampling_alpha(
                step,
                warmup_steps=self.history_schedule_warmup_steps,
                ramp_steps=self.history_schedule_ramp_steps,
                max_alpha=self.history_schedule_max_alpha,
            )
            mixed_history, history_mask = corrupt_history(
                history_rng,
                observation.action_history,
                observation.action_history_mask,
                predicted_history,
                prediction_mask=prediction_mask,
                noise_std=self.history_noise_std,
                dropout_probability=self.history_dropout_probability,
                alpha=alpha,
            )
            observation = observation.replace(
                action_history=mixed_history,
                action_history_mask=history_mask,
            )

        current_output = self.flow_forward(
            current_forward_rng,
            observation,
            actions,
            train=train,
            noise=current_noise,
            time=time,
        )
        first_order, second_order = continuity_penalties(current_output.endpoint)
        if self.boundary_weight > 0:
            boundary = boundary_consistency_loss(
                current_output.endpoint,
                previous_output.endpoint,
                overlap=self.overlap_O,
                current_target=actions,
                previous_target=previous_actions,
            )
        else:
            boundary = jnp.zeros((), dtype=actions.dtype)

        total, metrics = combine_supervised_losses(
            current_output.per_step_loss,
            first_order=first_order,
            second_order=second_order,
            boundary=boundary,
            first_weight=self.continuity_first_order_weight,
            second_weight=self.continuity_second_order_weight,
            boundary_weight=self.boundary_weight,
        )
        metrics = {
            **metrics,
            "loss/total": total,
            "history/alpha": alpha,
        }
        return total, metrics

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        # 仅使用前缀构建 KV cache，随后对后缀进行去噪积分
        observation = _model.preprocess_observation(
            None, observation, train=False)
        # 约定：t=1 为噪声，t=0 为目标（与论文记号相反，以扩散文献常用约定为准）
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(
                rng, (batch_size, self.action_horizon, self.action_dim))

        # 先前向一次前缀以填充 KV cache
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(
            observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # 后缀内部注意力掩码：[B, S_suffix, S_suffix]
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # 后缀到前缀的注意力可见矩阵：[B, S_suffix, S_prefix]
            prefix_attn_mask = einops.repeat(
                prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # 拼接得到对整段（前缀+后缀）的可见性：[B, S_suffix, S_prefix + S_suffix]
            full_attn_mask = jnp.concatenate(
                [prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # 后缀的 positions 应偏移到前缀长度之后
            positions = jnp.sum(
                prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            # 对浮点误差鲁棒的终止条件
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
