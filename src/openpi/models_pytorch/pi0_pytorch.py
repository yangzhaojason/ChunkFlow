"""
PI0Pytorch: OpenPI 项目中基于 PyTorch 的 PI0/PI05 模型实现。

主要职责：
- 负责图像与文本前缀编码（SigLIP + 语言嵌入）、状态/动作/时间后缀编码。
- 将前缀与后缀拼接后，送入 PaliGemma 主干和 Expert 分支进行推理/训练。
- 支持基于 diffusion-like 过程的动作去噪训练与逐步推理（Euler 步进）。
- 可选启用梯度检查点以节省显存。

注意：本文件尽可能添加了中文注释，便于快速理解整体数据流与关键张量形状。
"""

import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

# Gemma/PaliGemma 相关组件与配置
import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel

# 观测预处理（图像、文本、状态等）
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    # 在 CPU 上不支持 bfloat16，因此强制回退到 float32；float64 保持不变
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    # 为标量时间步生成正余弦位置编码：形状为 [B, dimension]
    # 使用指数插值在 [min_period, max_period] 内均匀覆盖频率
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError(
            "The time tensor is expected to be of shape `(batch_size, )`.")

    # 为避免在 CPU 上使用不安全 dtype，选择合适的浮点精度
    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension //
                              2, dtype=dtype, device=device)
    # 按照 fraction 指数插值得到周期序列
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi  # 频率缩放
    sin_input = scaling_factor[None, :] * time[:, None]  # 外积：B×(D/2)
    # 拼接 sin 与 cos 通道，得到 [B, D]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    # 从 Beta(alpha, beta) 分布采样，常用于时间步/权重的随机化
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    # 基于 att_masks 的累积和，生成二维注意力可见性矩阵：
    # - att_2d_masks[b, i, j] 为 True 表示 token i 可以看到 token j
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    # pad_2d_masks 限制在 padding 内部不可见
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05  # 是否使用 PI05（含 adaRMS 时间条件）

        # 获取 PaliGemma 主干与 Expert 分支的配置
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        # 搭建含 Expert 的 PaliGemma 模型，PI05 时启用第二分支的 adaRMS
        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        # 动作通道（dim=32）到 Expert 宽度的投影，以及反向投影回动作空间
        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        if self.pi05:
            # PI05：时间条件通过一个 MLP（adaRMS 使用）
            self.time_mlp_in = nn.Linear(
                action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(
                action_expert_config.width, action_expert_config.width)
        else:
            # PI0：显式引入 state，并将 (action_emb, time_emb) 级联后过 MLP
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(
                2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(
                action_expert_config.width, action_expert_config.width)

        # 提升 matmul 精度以获得更稳定数值行为；对采样函数进行编译加速
        torch.set_float32_matmul_precision("high")
        self.sample_actions = torch.compile(
            self.sample_actions, mode="max-autotune")

        # 记录是否启用梯度检查点（可减少显存消耗）
        self.gradient_checkpointing_enabled = False

        # 运行时检查：确保替换后的 transformers 组件已正确安装
        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        # 同步设置主干与 Expert 的 checkpointing 标志
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        # 关闭梯度检查点
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        # 在训练且开启 checkpointing 时，对指定计算块进行重计算换显存
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        # 将 [B, N, N] 扩展为 [B, 1, N, N]，并用极小值屏蔽无效位置
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        # 统一进行图像归一化、tokenize、mask 等预处理
        observation = _preprocessing.preprocess_observation_pytorch(
            observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        # 采样标准高斯噪声，用于构建 x_t 与训练目标 u_t
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        # 从 Beta(1.5, 1.0) 采样时间步，并线性映射到 (0,1]
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前缀嵌入：将多路图像与语言 token 编码为 Transformer 前缀序列。

        参数：
        - images: list[Tensor]，每路图像形状 [B, C, H, W]
        - img_masks: list[BoolTensor]，每路图像有效性掩码 [B]
        - lang_tokens: LongTensor，语言 token 序列 [B, N_lang]
        - lang_masks: BoolTensor，语言 token 的有效位置 [B, N_lang]

        返回：
        - embs: FloatTensor，前缀嵌入 [B, N_prefix, D]
        - pad_masks: BoolTensor，padding 掩码 [B, N_prefix]，True 表示有效 token
        - att_masks: BoolTensor，可见性向量 [B, N_prefix]，0/1 控制该位置能否被“其他组”看到
          约定：图像之间互相可见（0），语言彼此也可见（0）；前后缀之间的互相可见性由后续拼接决定
        """
        # 前缀包含：若干图像 token（SigLIP 编码）与语言 token（嵌入后按 sqrt(d) 缩放）
        # 注意：此处仅构建前缀内部的自可见性，跨前缀/后缀的可见性在外层完成
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):
            # 对每个图像进行编码；img: [B, C, H, W]；img_mask: [B]

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens（语言 token 编码 + sqrt(d) 缩放）
        def lang_embed_func(lang_tokens):
            # 语言 token 嵌入，并按 sqrt(d) 缩放提升数值稳定
            lang_emb = self.paligemma_with_expert.embed_language_tokens(
                lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # 图像与语言之间允许完全注意力（互相可见）
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(
            att_masks, dtype=torch.bool, device=pad_masks.device)

        # 扩展 att_masks 到 batch 维度，使其成为 [B, N_prefix]
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """后缀嵌入：将状态、带噪动作与时间步编码为 Expert 分支输入。

        参数：
        - state: FloatTensor，连续状态 [B, A]（仅 PI0 使用；PI05 将状态离散化为 token 由前缀承担）
        - noisy_actions: FloatTensor，x_t（与噪声/动作线性混合）[B, T, A]
        - timestep: FloatTensor，时间步 t∈[0,1] [B]

        返回：
        - embs: FloatTensor，后缀嵌入 [B, N_suffix, D]
        - pad_masks: BoolTensor，后缀 padding 掩码 [B, N_suffix]
        - att_masks: FloatTensor，可见性向量 [B, N_suffix]（按外层约定构造 0/1 模式）
        - adarms_cond: FloatTensor | None，adaRMS 条件（仅 PI05 返回 [B, D]；PI0 为 None）

        说明：
        - PI0：state -> 线性投影；time -> 与 action_emb 级联后过小型 MLP；adarms_cond=None
        - PI05：state 不在此处编码；time -> 过时间 MLP 输出 adarms_cond；action_emb 直接作为后缀
        - att_masks：第一个后缀 token（state 或 action_t0）为 1（不可被前缀看到），
          后续 action token 设为 0（允许跨 action 内部注意力），具体可见性在全局掩码中拼接决定
        """
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            # PI0：包含 state，注意类型与权重 dtype 对齐
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        # 时间位置编码：敏感度范围 [0,1]，输出维度与 Expert 宽度对齐
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            # 动作投影到模型宽度空间
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            # PI0：将时间编码扩展到动作序列长度后与动作嵌入拼接，随后用小型 MLP 融合
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                # 小型 MLP 融合 action/time 信息
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # 将 action/time 融合后的序列加入后缀 token 序列
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(
            bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # 可见性：使图像/语言/状态（前缀部分）不去关注 action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(
            att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """训练前向，计算逐元素 MSE 损失（[B, T, A]）。

        参数：
        - observation: 统一观测结构，含多路图像/掩码、语言 token/掩码、连续状态；来自数据管道
        - actions: 真实动作序列，形状 [B, T, A]
        - noise: 可选外部噪声，若为 None 则内部采样，形状与 actions 相同
        - time: 可选外部时间步 t∈[0,1]，若为 None 则内部按批次采样，形状 [B]

        流程概述：
        1) 预处理观测，得到前缀（图像/语言）嵌入与掩码，后缀（状态+动作/时间）嵌入与掩码
        2) 构造噪声-动作的线性混合 x_t = t·noise + (1−t)·actions，并以 u_t = noise − actions 作为监督目标
        3) 将前缀/后缀拼接后送入组合模型（PaliGemma + Gemma expert），仅保留后缀对应输出
        4) 经线性层映射到动作维度，计算与 u_t 的逐元素均方误差
        """
        # 1) 预处理观测，拆分出图像/语言/状态张量及其掩码
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(
            observation, train=True)

        # 2) 若未给定，采样训练噪声 ~ N(0, I)，形状与 actions 相同
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        # 2) 若未给定，采样时间步 t∈[0,1]，形状 [B]
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        # 将时间步扩展以便与 [B, T, A] 广播；构造 x_t 与监督目标 u_t
        time_expanded = time[:, None, None]  # [B, 1, 1]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions  # 训练目标：噪声与真值动作之间的差

        # 3) 计算前缀（图像+语言）嵌入与掩码
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks)
        # 3) 计算后缀（状态+动作/时间）嵌入与掩码；并得到 adaRMS 条件（Pi05 时生效）
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            state, x_t, time)

        # 若语言模型权重为 bfloat16，则将嵌入也统一为 bfloat16 以避免 dtype 不匹配
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[
                0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            # 若主干为 bfloat16，则将拼接后的嵌入降精度以减少类型不匹配
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        # 将前缀/后缀的 padding 掩码与注意力可见向量拼接
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        # 生成二维注意力掩码 [B, S, S]，以及自增位置编码 position_ids [B, S]
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1  # 自然位置编码（跳过 padding）

        # 将二维掩码转换为 HF 所需的 4D 形状（广播到多头）
        # 形状约为 [B, 1, S_q, S_k]
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # 可选：在前向主干上应用梯度检查点以节省显存
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            # 执行一次无缓存的前向，返回后缀序列的隐表示
            # inputs_embeds = [前缀嵌入, 后缀嵌入]；只取后缀输出
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        # 以检查点包装主干前向（若已启用），减小峰值显存
        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        # 仅保留动作窗口对应的尾部 token（后缀长度与动作时域一致）
        suffix_out = suffix_out[:, -self.config.action_horizon:]
        # 为保持数值稳定，在线性映射前将隐表示转为 float32
        suffix_out = suffix_out.to(dtype=torch.float32)

        # 可选：在最终动作投影层也应用检查点
        def action_out_proj_func(suffix_out):
            # 将隐表示映射回动作空间（32 维）
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        # 返回逐元素 MSE，用于与调度/加权策略组合
        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """推理采样：给定观测，进行反向 SDE/Euler 积分得到动作序列。

        参数：
        - device: 设备（cuda/cpu）
        - observation: 观测，包含图像/语言/状态
        - noise: 初始噪声 x_{t=1}，[B, T, A]，若 None 则内部采样
        - num_steps: 反向积分步数（越大越细致，时间越长）

        返回：
        - actions: FloatTensor，去噪后的动作序列 [B, T, A]

        流程：
        1) 仅对前缀（图像+语言）做一次前向，构建 KV cache
        2) 从 t=1 反向到 t=0，循环执行 denoise_step 得到 v_t，并用 Euler 更新 x_t
        3) 最终返回 x_t 作为采样动作
        """
        # 推理阶段：给定观测与初始噪声，利用 Euler 反向积分逐步去噪，得到动作序列
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon,
                             self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(
            observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(
            prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache（缓存仅由前缀构建）
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(
            prefix_att_2d_masks)
        # 计算前缀的 KV cache，后续去噪步骤仅追加后缀
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # 反向时间积分步长（负号表示从 t=1 递减至 t=0）
        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        # 以固定步长积分，直到到达 t≈0（半步终止条件）
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step：x_{t+dt} = x_t + dt * v_t
            # 使用新的张量赋值，避免 inplace 带来的梯度/编译问题
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        # 单步去噪：仅对后缀执行前向，前缀通过 KV cache 参与注意力
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        # 构造后缀对前缀的注意力可见矩阵：[B, S_suffix, S_prefix]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(
            suffix_pad_masks, suffix_att_masks)

        # 拼接得到完整二维注意力掩码：[B, S_suffix, S_prefix + S_suffix]
        full_att_2d_masks = torch.cat(
            [prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        # 后缀的 position_ids 需要偏移到前缀长度之后
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + \
            torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(
            full_att_2d_masks)
        # Expert 分支使用 eager 注意力实现以配合 cache
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
