"""Pi0-FAST 模型定义（JAX/Flax NNX 版本）。

主要特性：
- 使用 Gemma（通过 bridge）作为语言主干，SigLIP 作为图像编码器
- 将图像嵌入与文本 token 嵌入按前缀-语言建模（prefix-lm）范式拼接
- 提供快速解码路径（prefill + while_loop decode）
"""

import dataclasses
import logging
from typing import Any

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma_fast as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

# 获取项目日志记录器
logger = logging.getLogger("openpi")

# PaliGemma 的 EOS token（用于解码/提前停止）
PALIGEMMA_EOS_TOKEN = 1


def make_attn_mask(input_mask, mask_ar):
    """从 big_vision 改编：根据自回归掩码构造前缀/因果注意力。

    语义：token 只能关注到“累计 mask_ar 不大于自身”的位置；这样就能用 bool[?B,N] 的 `mask_ar`
    统一描述不同的注意力策略，例如：
      - [[1 1 1 1 1 1]]：纯因果注意力（只能看过去）
      - [[0 0 0 1 1 1]]：prefix-lm（前三个互看，后三个因果）
      - [[1 0 1 0 1 0 0 1 0 0]]：按块的因果注意（每块内互看，能看之前所有块）

    参数：
    - input_mask: bool[B, N]，True 表示有效 token，False 为 padding
    - mask_ar: bool[?B, N]，True 表示该位置开始一个新的“注意力段”，False 表示延续之前的段
    """
    # 将 mask_ar 广播到 batch
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    # 对每个位置计算所属段的累计编号
    cumsum = jnp.cumsum(mask_ar, axis=1)
    # 仅允许关注到“段编号不大于自身”的位置（段内全可见，段间因果）
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    # 仅在有效 token 区域内生效
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@jax.vmap
def left_to_right_align(x, input_mask, attn_mask):
    """将左对齐序列转换为右对齐（便于解码时以右端为最近位置）。

    由于使用 vmap，本函数在单样本维度上运行。
    """
    # 形状与一致性检查
    assert x.ndim == 2
    assert input_mask.ndim == 1
    assert attn_mask.ndim == 2
    assert x.shape[0] == input_mask.shape[0]
    assert attn_mask.shape[0] == attn_mask.shape[1], attn_mask.shape
    # 计算有效长度，并滚动使有效段移至序列末尾（右对齐）
    seqlen = jnp.max(input_mask * jnp.arange(input_mask.shape[0])) + 1
    x = jnp.roll(x, -seqlen, axis=0)
    input_mask = jnp.roll(input_mask, -seqlen, axis=0)
    attn_mask = jnp.roll(attn_mask, -seqlen, axis=(0, 1))
    return x, input_mask, attn_mask


def put_along_last_axis(arr, indices, values):
    """等价于 np.put_along_axis(..., axis=-1) 的 JAX 实现（JAX 无此 API）。"""
    assert arr.ndim == indices.ndim == values.ndim, (
        arr.ndim, indices.ndim, values.ndim)
    onehot = jax.nn.one_hot(indices, arr.shape[-1], dtype=values.dtype)
    put_mask = jnp.einsum("...i,...in->...n",
                          jnp.ones(values.shape, jnp.int32), onehot)
    put_values = jnp.einsum("...i,...in->...n", values, onehot)
    return jnp.where(put_mask, put_values, arr)


@dataclasses.dataclass(frozen=True)
class Pi0FASTConfig(_model.BaseModelConfig):
    # 计算精度（字符串），例如 "bfloat16"
    dtype: str = "bfloat16"
    # PaliGemma/Gemma 主干变体（快速版）
    paligemma_variant: _gemma.Variant = "gemma_2b"

    # Set the model specific defaults.
    # 动作维度与时域
    action_dim: int = 32
    action_horizon: int = 32
    # 最大语言 token 长度
    max_token_len: int = 250

    # Tokenizer for the fast model.
    # 可选的快速模型 tokenizer（用于外部注入）
    fast_model_tokenizer: Any | None = None
    # Keyword arguments for the fast model tokenizer.
    # tokenizer 的 kwargs
    fast_model_tokenizer_kwargs: dict[str, Any] | None = None

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0_FAST

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0FAST":
        return Pi0FAST(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        # 定义图像与掩码的规格
        image_spec = jax.ShapeDtypeStruct(
            [batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            # 观测规格：三路图像、语言 token/掩码、自回归掩码、loss 掩码等
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "base_1_rgb": image_spec,
                    "wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "base_1_rgb": image_mask_spec,
                    "wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct(
                    [batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct(
                    [batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct(
                    [batch_size, self.max_token_len], bool),
                token_ar_mask=jax.ShapeDtypeStruct(
                    [batch_size, self.max_token_len], jnp.int32),
                token_loss_mask=jax.ShapeDtypeStruct(
                    [batch_size, self.max_token_len], jnp.bool_),
            )
        # 动作规格：[B, T, A]
        action_spec = jax.ShapeDtypeStruct(
            [batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """根据配置返回参数冻结规则：若使用 LoRA 变体，冻结 LLM 非 LoRA 参数。"""
        if "lora" in self.paligemma_variant:
            return nnx.All(nnx_utils.PathRegex(".*llm.*"), nnx.Not(nnx_utils.PathRegex(".*lora.*")))
        return nnx.Nothing


class Pi0FAST(_model.BaseModel):
    def __init__(self, config: Pi0FASTConfig, rngs: nnx.Rngs):
        # 初始化父类（指定动作维度/时域/最大 token 长度）
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        # 获取 Gemma 快速变体配置
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        # TODO: 未来将 Gemma 重写为原生 NNX；当前通过 bridge 转接
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                **paligemma_config,
                embed_dtype=config.dtype,
                cache_dtype=config.dtype,
            )
        )
        # 延迟初始化 LLM（节省编译/构图开销）
        llm.lazy_init(rngs=rngs, method="init")
        # SigLIP 图像编码器，同样通过 bridge 转接为 NNX
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        # 以假观测触发 image 编码器的 lazy_init
        img.lazy_init(next(iter(config.fake_obs().images.values())),
                      train=False, rngs=rngs)
        # 聚合子模块
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

    @at.typecheck
    def embed_inputs(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Int[at.Array, "b s"]]:
        """将多路图像与文本 token 编码并拼接为一段输入序列。

        返回：
        - token_embeddings: [B, S, D]
        - input_mask: [B, S]，有效位置
        - ar_mask: [B, S]，自回归段落划分（0=延续，1=新段）
        """
        input_mask = []
        ar_mask = []
        token_embeddings = []
        # 嵌入多路图像
        for name in obs.images:
            image_token_embeddings, _ = self.PaliGemma.img(
                obs.images[name], train=False)

            token_embeddings.append(image_token_embeddings)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_token_embeddings.shape[1],
                )
            )
            # 图像 token 彼此可见（同一段）--> ar_mask=0
            ar_mask.append(0 * input_mask[-1])

        # 添加文本 token（要求上游已提供 token、mask、ar mask）
        assert obs.tokenized_prompt is not None, "Tokenized prompt is required"
        assert obs.tokenized_prompt_mask is not None, "Tokenized prompt mask is required"
        assert obs.token_ar_mask is not None, "Token auto-regressive mask is required"
        tokenized_inputs_embeddings = self.PaliGemma.llm(
            obs.tokenized_prompt, embed_only=True)
        token_embeddings.append(tokenized_inputs_embeddings)
        input_mask.append(obs.tokenized_prompt_mask)
        ar_mask.append(obs.token_ar_mask)

        # 拼接得到完整序列及其掩码
        return (
            jnp.concatenate(token_embeddings, axis=1),
            jnp.concatenate(input_mask, axis=1),
            jnp.concatenate(ar_mask, axis=1),
        )

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        # 统一预处理观测（归一化/裁剪/对齐图像键等）
        observation = _model.preprocess_observation(
            rng, observation, train=train, image_keys=list(
                observation.images.keys())
        )

        # 组合前缀与语言序列，一次性前向
        input_token_embeddings, input_mask, ar_mask = self.embed_inputs(
            observation)
        attn_mask = make_attn_mask(input_mask, ar_mask)

        # 构造 one-hot 目标：预测“下一个 token”，因此目标为 prompt 从索引 1 开始
        targets = jax.nn.one_hot(
            observation.tokenized_prompt[:, 1:],
            self.PaliGemma.llm.module.vocab_size,
        )

        # 每个输入位置预测下一个 token，因此输入端去掉最后一个位置
        pre_logits, _, _ = self.PaliGemma.llm(
            embedded_prefix=input_token_embeddings[:, :-1],
            mask=attn_mask[:, :-1, :-1],
            return_prelogits=True,
        )

        # 仅对目标长度解码 logits 以节省显存（seq_len × vocab 的大矩阵乘）
        logits, _ = self.PaliGemma.llm(
            pre_logits=pre_logits[:, -targets.shape[1]:],
        )
        logp = jax.nn.log_softmax(logits, axis=-1)

        # 计算交叉熵损失（按 token_loss_mask 位置统计）
        assert observation.token_loss_mask is not None, "Token loss mask is required"
        loss_mask = observation.token_loss_mask[:, 1:]
        token_pplx = jnp.sum(targets * logp, axis=-1)
        return -jnp.sum(token_pplx * loss_mask, axis=-1) / jnp.clip(jnp.sum(loss_mask, -1), 1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        max_decoding_steps: int | at.Int[at.Array, ""] = 256,
        temperature: float = 0.0,
    ) -> _model.Actions:
        # TODO：目前通过 preprocess_observation 取出图像键，后续可以合并接口
        observation = _model.preprocess_observation(
            None, observation, train=False, image_keys=list(observation.images.keys())
        )

        # 前缀嵌入并构造前缀注意力掩码
        prefix_token_embeddings, prefix_mask, prefix_ar_mask = self.embed_inputs(
            observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)

        # 将所有输入序列右对齐，保证最新 token 位于右端
        prefix_token_embeddings, prefix_mask, prefix_attn_mask = left_to_right_align(
            prefix_token_embeddings, prefix_mask, prefix_attn_mask
        )
        prefill_size = prefix_token_embeddings.shape[1]
        prefill_len = jnp.sum(prefix_mask, axis=-1)
        prefix_start = prefill_size - prefill_len

        # 先用前缀一次前向填充 KV cache；将注意力 mask 按最大步数右侧填充，确定 cache 大小
        prefix_attn_mask = jnp.pad(
            prefix_attn_mask, ((0, 0), (0, 0), (0, max_decoding_steps)))
        prefix_positions = jnp.cumsum(prefix_mask, axis=-1) - 1
        prefix_logits, kv_cache, _ = self.PaliGemma.llm(
            embedded_prefix=prefix_token_embeddings, mask=prefix_attn_mask, positions=prefix_positions, decode=True
        )

        # 解码准备：prefix 的最后一个 logit 将用于解出第一个 token
        last_logit = prefix_logits[:, -1:]
        output_tokens = jnp.zeros((last_logit.shape[0], max_decoding_steps))

        def step(carry):
            rng, last_logit, output_tokens, cache, _, step = carry

            # 从最后一个 logit 采样/贪心一个 token，并写入输出的当前位置
            # 为当前步划分 RNG
            rng, rng_step = jax.random.split(rng)
            token = jax.lax.cond(
                temperature > 0.0,
                lambda _: jax.random.categorical(
                    rng_step, last_logit / temperature, axis=-1),
                lambda _: jnp.argmax(last_logit, axis=-1),
                operand=None,
            )
            output_tokens = put_along_last_axis(
                output_tokens, jnp.broadcast_to(step, (token.shape[0], 1)), token)

            # 提前停止：若全 batch 均采到 EOS，则终止
            has_eos = jnp.any(token == PALIGEMMA_EOS_TOKEN, axis=-1)
            all_eos = jnp.all(has_eos)

            # 解码一步：将采样到的 token 嵌入，按当前位置更新 mask 与 positions，并继续前向
            token_embedding = self.PaliGemma.llm(token, embed_only=True)
            positions = prefill_len[:, None] + step + 1
            mask = jnp.logical_and(
                jnp.arange(
                    prefill_size + max_decoding_steps)[None, None, :] >= prefix_start[:, None, None],
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :]
                < (jnp.broadcast_to(prefill_size + step + 1, (prefix_start.shape[0], 1, 1))),
            )
            last_logit, kv_cache, _ = self.PaliGemma.llm(
                embedded_prefix=token_embedding, mask=mask, positions=positions, decode=True, kv_cache=cache
            )

            return rng, last_logit, output_tokens, kv_cache, all_eos, step + 1

        def cond(carry):
            _, _, _, _, all_eos, step = carry
            return (~all_eos) & (step < max_decoding_steps)

        # 使用 lax.while_loop 以便将解码循环整体 jit，提升速度
        _, _, output_tokens, _, _, _ = jax.lax.while_loop(
            cond, step, (rng, last_logit, output_tokens, kv_cache, False, 0)
        )
        return output_tokens
