"""JAX 版训练主脚本。

功能：
- 初始化日志与 wandb（可恢复）
- 构建数据加载器与初始批，记录可视化样例
- 初始化模型/优化器/训练状态，支持从权重加载与断点恢复
- jit 并分片 train_step，执行训练循环、日志与保存 checkpoint
"""

import dataclasses  # 标准库：数据类工具
import functools  # 标准库：函数工具（偏函数等）
import logging  # 标准库：日志系统
import platform  # 标准库：平台与主机信息
from typing import Any  # 类型注解：任意类型

import etils.epath as epath  # 第三方：更友好的 Path 接口
import flax.nnx as nnx  # Flax NNX：状态化模块系统
from flax.training import common_utils  # Flax 训练常用工具
import flax.traverse_util as traverse_util  # Flax pytree 遍历工具
import jax  # JAX 主库
import jax.experimental  # JAX 实验特性（保持导入以注册功能）
import jax.numpy as jnp  # JAX 数组 API（NumPy 风格）
import numpy as np  # NumPy（用于 CPU 端拼接/日志）
import optax  # 优化器库
import tqdm_loggable.auto as tqdm  # 可记录到日志的 tqdm 进度条
import wandb  # Weights & Biases 日志

from openpi.models import chunkflow_critic as _chunkflow_critic
import openpi.models.model as _model  # 项目：模型基类与类型
import openpi.shared.array_typing as at  # 项目：数组/类型别名
import openpi.shared.nnx_utils as nnx_utils  # 项目：NNX 工具函数
from openpi.training import chunkflow_awac_train as _chunkflow_awac_train
from openpi.training import chunkflow_batch as _chunkflow_batch
from openpi.training import chunkflow_train as _chunkflow_train
import openpi.training.checkpoints as _checkpoints  # 项目：checkpoint 管理
import openpi.training.config as _config  # 项目：训练配置
import openpi.training.data_loader as _data_loader  # 项目：数据加载器
import openpi.training.optimizer as _optimizer  # 项目：优化器/学习率调度
import openpi.training.sharding as sharding  # 项目：分片与并行策略
import openpi.training.utils as training_utils  # 项目：训练通用工具
import openpi.training.weight_loaders as _weight_loaders  # 项目：权重加载器


def init_logging():  # 初始化日志格式与等级
    """自定义日志格式，提升可读性。"""  # 函数说明：设置日志格式
    level_mapping = {"DEBUG": "D", "INFO": "I",
                     "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}  # 日志等级缩写映射

    class CustomFormatter(logging.Formatter):  # 自定义格式化器类
        def format(self, record):  # 覆写格式化逻辑
            record.levelname = level_mapping.get(
                record.levelname, record.levelname)  # 将等级名替换为短字母
            return super().format(record)  # 调用父类进行最终格式化

    formatter = CustomFormatter(
        # 日志行格式
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",  # 时间显示格式
    )

    logger = logging.getLogger()  # 获取根 logger
    logger.setLevel(logging.INFO)  # 设置日志级别为 INFO
    logger.handlers[0].setFormatter(formatter)  # 替换默认 handler 的格式化器


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):  # 初始化 wandb
    """初始化 wandb 运行，支持恢复与可选代码归档。"""  # 说明：可选择记录代码快照
    if not enabled:  # 若禁用 wandb
        wandb.init(mode="disabled")  # 以禁用模式初始化（不产生外部通信）
        return  # 提前返回

    ckpt_dir = config.checkpoint_dir  # checkpoint 根目录
    if not ckpt_dir.exists():  # 目录必须存在
        raise FileNotFoundError(
            f"Checkpoint directory {ckpt_dir} does not exist.")  # 报错：目录不存在
    if resuming:  # 若为恢复运行
        # 读取先前保存的 run id
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(mode="offline", id=run_id, resume="must",
                   project=config.project_name)  # 离线恢复到同一 run
    else:  # 否则创建新 run
        wandb.init(
            mode="offline",  # 采用离线模式（可后续同步）
            name=config.exp_name,  # 运行名称
            config=dataclasses.asdict(config),  # 将配置字典记录到 run
            project=config.project_name,  # 项目名
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)  # 写回 run id 便于恢复

    if log_code:  # 若需要记录代码
        # 归档代码目录至 wandb（可选）
        wandb.run.log_code(epath.Path(__file__).parent.parent)  # 记录上级目录中的代码


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:  # 加载并校验权重
    """加载并校验权重，返回与模型形状匹配的子集。"""  # 确保权重结构/shape/dtype 与期望一致
    loaded_params = loader.load(params_shape)  # 按期望的形状信息进行加载

    # Remove jax.ShapeDtypeStruct from the loaded params first
    loaded_params_clean = traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(
            loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )

    # Check if structures match
    loaded_keys = set(traverse_util.flatten_dict(loaded_params_clean).keys())
    expected_keys = set(traverse_util.flatten_dict(params_shape).keys())

    if loaded_keys != expected_keys:
        missing_in_loaded = expected_keys - loaded_keys
        extra_in_loaded = loaded_keys - expected_keys
        if missing_in_loaded:
            logging.warning(f"Loading partial parameters. New parameters (will be randomly initialized): {missing_in_loaded}")
        if extra_in_loaded:
            logging.warning(f"Checkpoint has extra parameters that will be ignored: {extra_in_loaded}")
        # Return only the intersection - new params will stay randomly initialized
        return loaded_params_clean

    # Full match: validate shapes and dtypes
    at.check_pytree_equality(
        expected=params_shape, got=loaded_params_clean, check_shapes=True, check_dtypes=True)

    return loaded_params_clean


@at.typecheck  # 类型检查：保证接口一致
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:  # 返回训练状态与分片信息
    """创建训练状态（模型/优化器/参数），或在恢复模式下返回形状与分片信息。"""
    tx = _optimizer.create_optimizer(
        config.optimizer, config.lr_schedule, weight_decay_mask=None)  # 构建优化器与学习率计划

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:  # 内部初始化函数
        critic_rng, model_rng = jax.random.split(rng)  # 拆分随机数键
        # 初始化模型与参数
        model = config.model.create(model_rng)  # 根据配置创建模型实例

        # 将部分权重混入模型（仅替换子集）
        if partial_params is not None:  # 如果提供了外部加载的部分参数
            graphdef, state = nnx.split(model)  # 拆分图结构与状态
            # Use intersect to only load parameters that exist in both checkpoint and model
            # This allows loading from checkpoints that don't have new parameters (e.g., Q/V heads)
            partial_params_filtered = traverse_util.unflatten_dict({
                k: v for k, v in traverse_util.flatten_dict(partial_params).items()
                if k in traverse_util.flatten_dict(state.to_pure_dict())
            })
            state.replace_by_pure_dict(partial_params_filtered)  # 用部分参数覆盖相应子树
            model = nnx.merge(graphdef, state)  # 合并回模型对象

        params = nnx.state(model)  # 导出全量状态（参数）
        # 将冻结参数转换为 bfloat16 以省显存
        params = nnx_utils.state_map(
            params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))  # 冻结参数降精度

        critic_params = critic_model_def = critic_opt_state = None
        target_v_params = target_v_model_def = reference_params = None
        if bool(getattr(config.model, "awac_enable", False)):
            critic = _chunkflow_critic.ChunkFlowCritic(
                state_dim=model.critic_state_dim,
                action_dim=config.model.action_dim,
                hidden_width=config.model.awac_critic_hidden_width,
                hidden_depth=config.model.awac_critic_depth,
                rngs=nnx.Rngs(critic_rng),
            )
            target_v_model_def = nnx.graphdef(critic.v)
            target_v_params = jax.tree.map(lambda value: value, nnx.state(critic.v))
            critic_model_def, critic_params = nnx.split(critic)
            critic_opt_state = tx.init(critic_params)
            reference_params = jax.tree.map(lambda value: value, params)

        return training_utils.TrainState(
            step=0,  # 初始步数
            params=params,  # 全量参数状态
            model_def=nnx.graphdef(model),  # 纯图定义（无参数）
            tx=tx,  # 优化器
            opt_state=tx.init(params.filter(
                config.trainable_filter)),  # 初始化可训练参数的优化器状态
            ema_decay=config.ema_decay,  # EMA 衰减系数
            ema_params=None if config.ema_decay is None else params,  # EMA 初值（如启用）
            critic_params=critic_params,
            critic_model_def=critic_model_def,
            critic_opt_state=critic_opt_state,
            target_v_params=target_v_params,
            target_v_model_def=target_v_model_def,
            reference_params=reference_params,
        )

    # 计算初始化形状与分片策略（恢复模式直接用此信息加载）
    train_state_shape = jax.eval_shape(init, init_rng)  # 静态求形状而不真正构造
    state_sharding = sharding.fsdp_sharding(
        train_state_shape, mesh, log=True)  # 基于 FSDP 的分片策略

    if resume:  # 恢复模式直接返回形状与分片
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(
        config.weight_loader, train_state_shape.params.to_pure_dict())  # 按形状加载部分权重
    replicated_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec())  # 复制分片（不切分）

    # 初始化训练状态并混入部分权重
    train_state = jax.jit(
        init,  # jit 编译初始化函数
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,  # 输入分片策略
        out_shardings=state_sharding,  # 输出分片策略
    )(init_rng, partial_params)  # 执行初始化

    _checkpoints.validate_awac_state(
        train_state,
        awac_enabled=bool(getattr(config.model, "awac_enable", False)),
    )

    return train_state, state_sharding  # 返回状态与分片


@at.typecheck  # 类型检查：训练步
def train_step(
    config: _config.TrainConfig,  # 训练配置
    rng: at.KeyArrayLike,  # 随机数键
    state: training_utils.TrainState,  # 当前训练状态
    batch: _data_loader.TrainingBatch,  # legacy 或 episode-paired batch
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:  # 返回新状态与指标字典
    """单步训练：前向计算损失，求梯度并应用更新，返回新状态与指标。"""
    is_awac_batch = isinstance(batch, _chunkflow_batch.ChunkFlowTrainBatch)
    awac_enabled = bool(getattr(config.model, "awac_enable", False))
    if awac_enabled and not is_awac_batch:
        raise ValueError("ChunkFlow AWAC requires a composite supervised/transition batch")
    if is_awac_batch and not awac_enabled:
        raise ValueError("received a ChunkFlow AWAC composite batch while AWAC is disabled")
    if awac_enabled:
        return _chunkflow_awac_train.train_step(config, rng, state, batch)

    is_paired_batch = isinstance(batch, _chunkflow_batch.PairedChunkBatch)
    chunkflow_enabled = bool(getattr(config.model, "chunkflow_supervised_enabled", False))
    if chunkflow_enabled and not is_paired_batch:
        raise ValueError("ChunkFlow supervision requires an episode-paired batch")
    if is_paired_batch and not chunkflow_enabled:
        raise ValueError("received a paired batch while ChunkFlow supervision is disabled")

    model = nnx.merge(state.model_def, state.params)  # 将图定义与参数合并为可训练模型
    model.train()  # 切换为训练模式（影响 Dropout/BN 等）

    def loss_fn(model: _model.BaseModel, rng: at.KeyArrayLike, train_batch, ema_model):
        if isinstance(train_batch, _chunkflow_batch.PairedChunkBatch):
            return model.compute_paired_loss(
                rng,
                train_batch.previous_observation,
                train_batch.previous_actions,
                train_batch.observation,
                train_batch.actions,
                ema_model=ema_model,
                step=train_batch.step,
                train=True,
            )
        observation, actions = train_batch
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss), {}

    train_rng = jax.random.fold_in(rng, state.step)  # 将步数折叠进随机键，保证每步随机性
    batch = _chunkflow_train.batch_with_step(batch, state.step)
    ema_model = None
    if isinstance(batch, _chunkflow_batch.PairedChunkBatch):
        if config.model.history_length > 0 and state.ema_params is None:
            raise ValueError("ChunkFlow history conditioning requires TrainConfig.ema_decay")
        if state.ema_params is not None:
            ema_model = nnx.merge(state.model_def, state.ema_params)
            ema_model.eval()

    # 仅对可训练参数求导与更新
    diff_state = nnx.DiffState(0, config.trainable_filter)  # 指定可微分参数子集
    (loss, loss_metrics), grads = nnx.value_and_grad(
        loss_fn,
        argnums=diff_state,
        has_aux=True,
    )(model, train_rng, batch, ema_model)

    params = state.params.filter(config.trainable_filter)  # 仅抽取可训练参数
    updates, new_opt_state = state.tx.update(
        grads, state.opt_state, params)  # 基于梯度计算参数更新
    new_params = optax.apply_updates(params, updates)  # 应用更新得到新参数

    # 就地更新模型参数并返回完整新状态
    nnx.update(model, new_params)  # 将新参数写回模型
    new_params = nnx.state(model)  # 导出全量状态（包含未训练部分）

    new_state = dataclasses.replace(
        state, step=state.step + 1, params=new_params, opt_state=new_opt_state)  # 步数+1 并更新状态
    if state.ema_decay is not None:  # 若启用 EMA
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old +
                (1 - state.ema_decay) * new, state.ema_params, new_params
            ),  # EMA 更新：加权旧值与新值
        )

    # 仅统计 kernel 参数的范数（排除偏置/scale/位置嵌入等一维参数）
    kernel_params = nnx.state(
        model,  # 在模型状态中筛选
        nnx.All(
            nnx.Param,  # 只看参数节点
            nnx.Not(nnx_utils.PathRegex(
                ".*/(bias|scale|pos_embedding|input_embedding)")),  # 排除某些一维/特定命名参数
            lambda _, x: x.value.ndim > 1,  # 仅保留多维权重（kernel）
        ),
    )
    info = {  # 记录训练相关指标
        "loss": loss,  # 损失
        "grad_norm": optax.global_norm(grads),  # 梯度全局范数
        "param_norm": optax.global_norm(kernel_params),  # 权重（kernel）全局范数
        **loss_metrics,
    }
    return new_state, info  # 返回新状态与指标


def main(config: _config.TrainConfig):  # 主函数：训练入口
    """训练入口：初始化、数据与状态创建，训练循环与保存。"""  # 高层流程说明
    init_logging()  # 设置日志格式
    logging.info(f"Running on: {platform.node()}")  # 打印当前主机名

    if config.batch_size % jax.device_count() != 0:  # 批大小需整除设备数
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )  # 避免数据分片不均

    jax.config.update("jax_compilation_cache_dir", str(
        epath.Path("~/.cache/jax").expanduser()))  # 设置 JAX 编译缓存目录

    rng = jax.random.key(config.seed)  # 基于种子创建随机键
    train_rng, init_rng = jax.random.split(rng)  # 切分为训练与初始化两个 RNG

    mesh = sharding.make_mesh(config.fsdp_devices)  # 构造设备 mesh
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))  # 数据按 batch 维分片
    replicated_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec())  # 参数复制到所有设备

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,  # checkpoint 根目录
        keep_period=config.keep_period,  # 长期保留周期
        overwrite=config.overwrite,  # 是否允许覆盖
        resume=config.resume,  # 是否从 checkpoint 恢复
    )  # 返回 checkpoint 管理器及是否处于恢复模式
    init_wandb(config, resuming=resuming,
               enabled=config.wandb_enabled)  # 初始化 wandb

    data_loader = _data_loader.create_data_loader(
        config,  # 数据相关配置
        sharding=data_sharding,  # 指定数据分片
        shuffle=True,  # 打乱数据
    )  # 构建数据加载器
    data_iter = iter(data_loader)  # 构造迭代器
    batch = next(data_iter)  # 预取一个 batch 用于形状推断/可视化
    logging.info(
        f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")  # 打印 batch 信息

    # Log images from first batch to sanity check.
    first_observation = _chunkflow_train.batch_observation(batch)
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i])
                    for img in first_observation.images.values()], axis=1))  # 拼接多相机图像
        # 最多可视化 5 张
        for i in range(min(5, len(next(iter(first_observation.images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)  # 在 step 0 记录图像

    # ---- RL signals quick check: rewards/discounts statistics (if present) ----
    try:
        obs0 = first_observation
        if hasattr(obs0, "rewards") and obs0.rewards is not None:
            rew = obs0.rewards
            rew_mean = float(jnp.mean(rew))
            rew_min = float(jnp.min(rew))
            rew_max = float(jnp.max(rew))
            rew_nonzero_ratio = float(jnp.mean(jnp.where(rew != 0, 1.0, 0.0)))

            if hasattr(obs0, "discounts") and obs0.discounts is not None:
                dsc = obs0.discounts
                dsc_mean = float(jnp.mean(dsc))
                dsc_min = float(jnp.min(dsc))
                dsc_max = float(jnp.max(dsc))
            else:
                dsc = None
                dsc_mean = dsc_min = dsc_max = float("nan")

            logging.info(
                "RL signals: rewards shape=%s, mean=%.6f, min=%.6f, max=%.6f, nonzero_ratio=%.3f; discounts mean=%.6f, min=%.6f, max=%.6f",
                str(rew.shape), rew_mean, rew_min, rew_max, rew_nonzero_ratio, dsc_mean, dsc_min, dsc_max,
            )
            wandb.log({
                "rl/rewards_mean": rew_mean,
                "rl/rewards_min": rew_min,
                "rl/rewards_max": rew_max,
                "rl/rewards_nonzero_ratio": rew_nonzero_ratio,
                "rl/discounts_mean": dsc_mean,
                "rl/discounts_min": dsc_min,
                "rl/discounts_max": dsc_max,
            }, step=0)
        else:
            logging.info("RL signals: rewards not present in batch[0]; AWAC will be inert unless provided.")
    except Exception as e:
        logging.warning(f"RL signals quick check failed: {e}")

    train_state, train_state_sharding = init_train_state(
        config, init_rng, mesh, resume=resuming)  # 初始化或准备状态形状
    jax.block_until_ready(train_state)  # 等待初始化完成
    logging.info(
        f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")  # 打印参数树信息

    if resuming:  # 若恢复，则还原状态（包含数据迭代器位置）
        train_state = _checkpoints.restore_state(
            checkpoint_manager, train_state, data_loader)
        _checkpoints.validate_awac_state(
            train_state,
            awac_enabled=bool(getattr(config.model, "awac_enable", False)),
        )

    ptrain_step = jax.jit(
        functools.partial(train_step, config),  # 绑定配置，得到无 config 参数的函数
        in_shardings=(replicated_sharding,
                      train_state_sharding, data_sharding),  # 输入分片策略：R, state, data
        out_shardings=(train_state_sharding,
                       replicated_sharding),  # 输出分片策略：state, R
        donate_argnums=(1,),  # 捐赠 train_state，减少拷贝
    )  # 预编译训练步

    start_step = int(train_state.step)  # 起始 step（恢复时非 0）
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),  # 训练步范围
        initial=start_step,  # 初始位置
        total=config.num_train_steps,  # 总步数
        dynamic_ncols=True,  # 自适应列宽
    )  # 进度条

    infos = []  # 缓存若干步的指标用于平均
    for step in pbar:  # 训练主循环
        with sharding.set_mesh(mesh):  # 在 mesh 上下文进行 pjit 执行
            train_state, info = ptrain_step(
                train_rng, train_state, batch)  # 执行一步训练
        infos.append(info)  # 收集指标
        if step % config.log_interval == 0:  # 到日志步，聚合输出
            stacked_infos = common_utils.stack_forest(infos)  # 将多步指标按树状堆叠
            reduced_info = jax.device_get(
                jax.tree.map(jnp.mean, stacked_infos))  # 求均值并拉回 host
            info_str = ", ".join(f"{k}={v:.4f}" for k,
                                 v in reduced_info.items())  # 组装可读字符串
            pbar.write(f"Step {step}: {info_str}")  # 进度条写日志
            wandb.log(reduced_info, step=step)  # 同步到 wandb
            infos = []  # 清空缓存
        batch = next(data_iter)  # 取下一批

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:  # 保存 checkpoint 条件
            _checkpoints.save_state(
                checkpoint_manager, train_state, data_loader, step)  # 触发保存

    logging.info("Waiting for checkpoint manager to finish")  # 等待保存后台任务
    checkpoint_manager.wait_until_finished()  # 直至保存完成


if __name__ == "__main__":  # 脚本直接运行时
    main(_config.cli())  # 从命令行解析配置并启动训练
