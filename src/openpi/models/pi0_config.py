# 本文件定义 PI0/PI05 模型的配置类（基于 JAX/Flax NNX），
# 负责：
# - 指定模型的精度、主干变体（paligemma_variant）、动作专家变体（action_expert_variant）等
# - 定义动作维度、动作时域、最大 token 长度等模型超参数
# - 根据配置生成模型实例与输入规格
# - 提供参数冻结（freeze filter）的规则，便于微调/LoRA 冻结

# 导入 dataclasses，用于声明不可变的数据类配置
import dataclasses
# TYPE_CHECKING 仅在静态类型检查时可用，避免运行期开销
from typing import TYPE_CHECKING

# 引入 Flax NNX（新的 Flax API）以创建/管理模块与参数
import flax.nnx as nnx
# 引入 JAX 顶层模块
import jax
# 引入 JAX 的 numpy 变体
import jax.numpy as jnp
# 覆写标注（typing_extensions.override）用于标记子类实现覆盖父类方法
from typing_extensions import override

# 导入本项目的模型基类定义
from openpi.models import model as _model
# 导入 Gemma/PaliGemma 相关变体定义与工具
import openpi.models.gemma as _gemma
# 项目内共享的数组类型定义
from openpi.shared import array_typing as at
# NNX 工具函数（路径匹配等）
import openpi.shared.nnx_utils as nnx_utils

# 仅在类型检查时导入具体模型类型，避免运行时循环依赖
if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


# 使用冻结（frozen=True）的 dataclass，确保配置对象不可变
@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    # 计算精度（字符串形式），例如 "bfloat16"，影响参数与计算 dtype
    dtype: str = "bfloat16"
    # 视觉-语言主干（PaliGemma/Gemma）的变体选择，决定容量与参数规模
    paligemma_variant: _gemma.Variant = "gemma_2b"
    # 动作专家（动作解码/规划头）的变体选择，决定动作头容量
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    # 动作维度：单步动作向量的维度大小
    action_dim: int = 32
    # 动作时域：一次生成/监督的连续动作步数（序列长度）
    action_horizon: int = 50
    # 最大 token 长度：语言 token 的最大序列长度，None 表示按 pi05 与否在 __post_init__ 中设默认
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    # 是否启用 Pi05 变体：
    # - 将状态输入并入离散语言 token
    # - 动作专家使用 adaRMSNorm 注入 flow matching 时间步
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    # 是否将状态作为离散输入（由 ModelTransformFactory 读取，不直接被模型使用）
    discrete_state_input: bool = None  # type: ignore

    # ChunkFlow continuity regularization (no-RL):
    # First-order smoothness weight: encourages |a_t - a_{t-1}| to be small
    continuity_first_order_weight: float = 0.0
    # Second-order smoothness weight: encourages |a_t - 2 a_{t-1} + a_{t-2}|^2 to be small
    continuity_second_order_weight: float = 0.0

    # ChunkFlow-style RL regularization (simplified):
    # Self-consistency weight: predict twice with different noise, penalize disagreement
    self_consistency_weight: float = 0.0
    # Temperature for consistency loss (higher = more tolerant of differences)
    consistency_temperature: float = 1.0

    # AWAC-specific hyperparameters
    awac_enable: bool = False
    awac_gamma: float = 0.99
    awac_expectile_tau_e: float = 0.7
    awac_temperature_tau: float = 0.05
    awac_wmax: float = 20.0
    awac_actor_weight: float = 1.0
    awac_q_weight: float = 1.0
    awac_v_weight: float = 1.0
    kl_beta: float = 0.0
    entropy_lambda: float = 0.0

    # Boundary consistency regularization
    overlap_O: int = 0  # Overlap length for boundary consistency
    boundary_weight: float = 0.0

    def __post_init__(self):
        # 在 dataclass 初始化后补全默认值
        # 若 max_token_len 未显式给定：Pi05 默认 200，否则默认 48
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        # 若 discrete_state_input 未显式给定，默认为是否启用 Pi05
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        # 根据是否启用 Pi05 返回对应的模型类型枚举值
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        # 延迟导入具体模型实现，避免循环依赖
        from openpi.models.pi0 import Pi0

        # 使用给定随机种子构造 NNX 的随机源，并创建模型实例
        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        # 定义图像张量规格：[B, H, W, C]，dtype=float32
        image_spec = jax.ShapeDtypeStruct(
            [batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        # 定义图像 mask 规格：[B]，dtype=bool
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            # 组装观测（observation）规格：包含多路摄像头图像、对应 mask、状态向量与语言 token
            observation_spec = _model.Observation(
                images={
                    # 底座相机 RGB
                    "base_0_rgb": image_spec,
                    # 左腕相机 RGB
                    "left_wrist_0_rgb": image_spec,
                    # 右腕相机 RGB
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    # 各路图像的可见性/有效性 mask
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                # 连续状态向量：[B, action_dim]
                state=jax.ShapeDtypeStruct(
                    [batch_size, self.action_dim], jnp.float32),
                # 语言 token 序列：[B, max_token_len]
                tokenized_prompt=jax.ShapeDtypeStruct(
                    [batch_size, self.max_token_len], jnp.int32),
                # 语言 token 的有效位置 mask：[B, max_token_len]
                tokenized_prompt_mask=jax.ShapeDtypeStruct(
                    [batch_size, self.max_token_len], bool),
            )
        # 动作张量规格：[B, T, action_dim]，T=action_horizon
        action_spec = jax.ShapeDtypeStruct(
            [batch_size, self.action_horizon, self.action_dim], jnp.float32)

        # 返回观测与动作的规格元组
        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """根据配置返回参数冻结过滤器，用于微调时选择性冻结/解冻参数。
        逻辑要点：
        - 若 paligemma_variant 或 action_expert_variant 包含 "lora"，则识别出对应的 LLM 参数路径过滤器
        - 对仅冻结主干或仅冻结动作专家的情形进行路径包含/排除
        - 若任一处启用 LoRA，则统一排除所有 LoRA 参数（避免冻结它们）
        - 若最终未产生任何过滤规则，则返回 Nothing 表示不冻结
        """
        # 累积过滤器条件的列表
        filters = []
        # 标记是否启用过 LoRA 相关过滤
        has_lora = False
        # 匹配 Gemma 主干参数的路径规则（包含 llm）
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        # 匹配动作专家参数的路径规则（一般出现在 llm.*_1.*）
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        # 若主干变体包含 LoRA，则将主干参数加入冻结目标
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            # 如果只想冻结主干而非动作专家，则排除动作专家参数
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        # 若动作专家变体包含 LoRA，则将动作专家参数加入冻结目标
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        # 若任意位置使用 LoRA，则统一排除所有 LoRA 参数，避免将其冻结
        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        # 若没有任何过滤条件，返回 Nothing（不冻结）
        if not filters:
            return nnx.Nothing
        # 否则返回 AND 组合的过滤器
        return nnx.All(*filters)
