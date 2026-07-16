"""Pi0/Pi05 模型（JAX/Flax NNX）。

结构概览：
- VLM 主干：PaliGemma（通过 bridge 连接到 NNX）
- 动作专家：Gemma expert（与主干混合注意力）
- 图像编码：SigLIP（bridge 连接）
- 两种模式：Pi0（状态连续、无 adaRMS）与 Pi05（状态离散进 token，动作专家用 adaRMS 条件）
"""

import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

# 项目日志记录器
logger = logging.getLogger("openpi")


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

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        """将状态/动作/时间编码为后缀序列，返回嵌入、掩码以及 adaRMS 条件（Pi05）。"""
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
            ar_mask += [True]

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
        input_mask.append(
            jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # 图像/语言/状态不去关注 action tokens（第一位遮蔽，后续允许因果）
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        # 预处理观测；采样噪声与时间；构造 x_t 与监督目标 u_t
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(
            preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # 前缀+后缀 一次性前向
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(
            observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[
                None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])

        # Base flow-matching loss per-position (reduce over action dims)
        per_pos_mse = jnp.mean(jnp.square(v_t - u_t), axis=-1)  # [B, T]

        # ChunkFlow continuity regularization (no-RL):
        # We construct a proxy of predicted actions at the sampled time t as: a_hat = noise - v_t
        # and apply first-order (TV-like) and second-order smoothness penalties over the time axis.
        lam_tv = getattr(self, "continuity_first_order_weight", 0.0)
        lam_dd2 = getattr(self, "continuity_second_order_weight", 0.0)

        if lam_tv > 0.0 or lam_dd2 > 0.0:
            a_hat = noise - v_t  # [B, T, A]

            # First-order: mean L1 of temporal differences
            tv_loss = 0.0
            if lam_tv > 0.0:
                d1 = a_hat[:, 1:, :] - a_hat[:, :-1, :]
                tv_loss = jnp.mean(jnp.abs(d1), axis=(-1, -2))  # [B]

            # Second-order: mean L2 of temporal second differences
            dd2_loss = 0.0
            if lam_dd2 > 0.0 and self.action_horizon >= 3:
                d2 = a_hat[:, 2:, :] - 2 * a_hat[:, 1:-1, :] + a_hat[:, :-2, :]
                dd2_loss = jnp.mean(jnp.square(d2), axis=(-1, -2))  # [B]

            # Aggregate to per-sample scalar and return. Keep base per-position loss by averaging over time.
            base_per_sample = jnp.mean(per_pos_mse, axis=-1)  # [B]
            total_per_sample = base_per_sample + lam_tv * tv_loss + lam_dd2 * dd2_loss
            return total_per_sample

        # Default: return per-position loss (keeps existing behavior)
        return per_pos_mse

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
            x_t, time = carry
            # 对浮点误差鲁棒的终止条件
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
