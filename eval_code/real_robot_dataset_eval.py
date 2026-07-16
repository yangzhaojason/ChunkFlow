#!/usr/bin/env python3
"""Detailed real-robot RLDS evaluation for ChunkFlow checkpoints.

This script performs comprehensive offline evaluation of OpenPI models on an RLDS dataset:
1. Loads DROID dataset using LeRobot format
2. Uses OpenPI policy loading mechanism
3. Performs single-step and multi-step action predictions
4. Compares predictions with ground truth
5. Generates detailed evaluation metrics and visualizations
6. Supports task-based analysis and episode selection

Usage:
    uv run eval_code/real_robot_dataset_eval.py \
      --checkpoint-dir checkpoints/pi05_truth_cartesian_cloth/run/70000 \
      --dataset-path datasets/real_robot \
      --dataset-name chunkflow_real_cloth_v2 \
      --config pi05_truth_finetune_cartesian_cloth_new \
      --episode-start 10 --episode-end 15 \
      --output-dir outputs/real_robot/dataset_eval
"""

from openpi_client import image_tools
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config
from openpi.policies import policy as _policy
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm
import tyro
import torch.nn.functional as F
import torch
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import dataclasses
import json
import logging
import os
import statistics
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

# Set environment variables to avoid video processing issues
os.environ["LEROBOT_SKIP_VIDEO"] = "1"
os.environ["DISABLE_TORCHCODEC"] = "1"


# LeRobot imports

# OpenPI imports

# # Configure logging
# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s - %(levelname)s - %(message)s')
# print = logging.getLogger(__name__)

# Set matplotlib style
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial Unicode MS", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


@dataclasses.dataclass
class Args:
    """Arguments for the DROID dataset evaluation script."""

    #################################################################################################################
    # Dataset parameters
    #################################################################################################################
    # Path to DROID dataset
    dataset_path: str = "datasets/real_robot"
    dataset_name: str = "chunkflow_real_cloth_v2"
    episodes: Optional[List[int]] = None  # Specific episodes to evaluate
    episode_start: int = 0  # 起始episode索引
    episode_end: int = 10  # 结束episode索引（不包含）
    max_samples: Optional[int] = None  # Maximum number of samples to process
    tasks_filter: Optional[List[str]] = None  # Filter episodes by tasks
    chunk_size: int = 10  # 步数

    #################################################################################################################
    # Model parameters
    #################################################################################################################
    config: str = "pi05_truth_finetune_cartesian_cloth_new"  # OpenPI config name
    # Checkpoint directory
    checkpoint_dir: str = ""
    default_prompt: Optional[str] = None  # Default prompt for the model
    resize_size: int = 224  # Image resize size

    #################################################################################################################
    # Evaluation parameters
    #################################################################################################################
    multi_step_horizon: int = 10  # Number of steps for multi-step prediction
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # Device to run on
    # Skip video-related operations to avoid codec issues
    skip_video_processing: bool = True

    #################################################################################################################
    # Output parameters
    #################################################################################################################
    output_dir: str = "outputs/real_robot/dataset_eval"
    save_plots: bool = True  # Whether to save plots
    save_videos: bool = False  # Whether to save video visualizations
    detailed_logging: bool = True  # Whether to enable detailed logging


def _require_checkpoint_dir(checkpoint_dir: str) -> str:
    """Reject an omitted checkpoint before model or dataset initialization."""
    if not checkpoint_dir.strip():
        raise ValueError("--checkpoint-dir is required")
    return checkpoint_dir


class DROIDEvaluationMetrics:
    """Metrics collector for DROID evaluation."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.inference_times = []
        self.prediction_errors = []
        self.per_dim_errors = defaultdict(list)
        self.episode_metrics = []

    def add_inference_time(self, time_ms: float):
        """Add inference time measurement."""
        self.inference_times.append(time_ms)

    def add_prediction_error(self, error: float, per_dim_errors: List[float]):
        """Add prediction error measurements."""
        self.prediction_errors.append(error)
        for i, err in enumerate(per_dim_errors):
            self.per_dim_errors[i].append(err)

    def add_episode_metrics(self, episode_data: Dict):
        """Add episode-level metrics."""
        self.episode_metrics.append(episode_data)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "inference_time": {
                "mean": (
                    statistics.mean(
                        self.inference_times) if self.inference_times else 0
                ),
                "std": (
                    statistics.stdev(self.inference_times)
                    if len(self.inference_times) > 1
                    else 0
                ),
                "min": min(self.inference_times) if self.inference_times else 0,
                "max": max(self.inference_times) if self.inference_times else 0,
            },
            "prediction_error": {
                "mean": (
                    statistics.mean(self.prediction_errors)
                    if self.prediction_errors
                    else 0
                ),
                "std": (
                    statistics.stdev(self.prediction_errors)
                    if len(self.prediction_errors) > 1
                    else 0
                ),
                "min": min(self.prediction_errors) if self.prediction_errors else 0,
                "max": max(self.prediction_errors) if self.prediction_errors else 0,
            },
        }


class DROIDDatasetEvaluator:
    """Comprehensive DROID dataset evaluator using OpenPI models."""

    def __init__(self, args: Args):
        self.args = args
        self.dataset = None
        self.policy = None
        self.metrics = DROIDEvaluationMetrics()
        self.dataset_analysis = None
        self.episode_start = args.episode_start
        self.episode_end = args.episode_end
        self.tasks_filter = None  # Filter for tasks

        # Setup output directories
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)

        print(
            f"Initialized DROID evaluator with output dir: {self.output_dir}")

    def analyze_dataset_tasks(self) -> Dict:
        """分析数据集中的任务分布和episode长度（适配tfds RLDS格式）"""
        task_info = {}
        episode_info = {}
        task_episodes = defaultdict(list)
        task_lengths = defaultdict(list)

        # print("分析数据集任务分布...")

        # 遍历所有episode
        for ep_idx, episode in enumerate(self.dataset):
            steps = episode["steps"]
            # steps 是 tf.data.Dataset，不可下标
            steps_list = list(steps)  # 如果数据集很大，建议只取第一个step
            ep_length = len(steps_list)
            # print(f"Episode {ep_idx} 长度: {ep_length}")
            if ep_length == 0:
                continue
            first_step = steps_list[0]
            # print(f"first_step的key: {first_step.keys()}")
            # 你可以根据实际数据结构调整
            task_str = first_step.get("task", "") or first_step.get(
                "language_instruction", ""
            )
            # 如果是tf.Tensor，转为python字符串
            if hasattr(task_str, "numpy"):
                task_str = task_str.numpy()
            if isinstance(task_str, bytes):
                task_str = task_str.decode("utf-8")
            if not isinstance(task_str, str):
                task_str = str(task_str)
            if not task_str:
                task_str = "<空任务名称>"

            # 统计
            if task_str not in task_info:
                task_info[task_str] = {
                    "name": task_str,
                    "display_name": task_str,
                    "episodes": [],
                    "total_frames": 0,
                    "avg_length": 0,
                    "min_length": float("inf"),
                    "max_length": 0,
                    "lengths": [],
                }
            task_info[task_str]["episodes"].append(ep_idx)
            task_info[task_str]["total_frames"] += ep_length
            task_info[task_str]["min_length"] = min(
                task_info[task_str]["min_length"], ep_length
            )
            task_info[task_str]["max_length"] = max(
                task_info[task_str]["max_length"], ep_length
            )
            task_info[task_str]["lengths"].append(ep_length)

            episode_info[ep_idx] = {"length": ep_length, "tasks": [task_str]}
            task_episodes[task_str].append(ep_idx)
            task_lengths[task_str].append(ep_length)

        # 计算平均长度等统计
        for task_str in task_info:
            if task_info[task_str]["episodes"]:
                episode_count = len(task_info[task_str]["episodes"])
                task_info[task_str]["avg_length"] = (
                    task_info[task_str]["total_frames"] / episode_count
                )
                task_info[task_str]["episode_count"] = episode_count
                lengths = task_info[task_str]["lengths"]
                if lengths:
                    import numpy as np

                    task_info[task_str]["std_length"] = np.std(lengths)
                    task_info[task_str]["median_length"] = np.median(lengths)
                else:
                    task_info[task_str]["std_length"] = 0
                    task_info[task_str]["median_length"] = 0
            else:
                task_info[task_str]["min_length"] = 0
                task_info[task_str]["episode_count"] = 0
                task_info[task_str]["std_length"] = 0
                task_info[task_str]["median_length"] = 0

        print(f"共发现 {len(task_info)} 个任务。")
        return {
            "task_info": task_info,
            "episode_info": episode_info,
            "task_episodes": dict(task_episodes),
            "task_lengths": dict(task_lengths),
        }

    def select_episodes_for_evaluation(self) -> List[int]:
        """根据任务分析结果选择要评测的episodes"""

        if self.episode_end is None:
            # 如果没有指定结束episode，使用所有episodes
            selected_episodes = list(
                self.dataset_analysis["episode_info"].keys())
            print(f"未指定episode_end，将评测所有 {len(selected_episodes)} 个episodes")
            return selected_episodes

        task_info = self.dataset_analysis["task_info"]
        task_episodes = self.dataset_analysis["task_episodes"]

        # 如果指定了任务过滤器
        if self.tasks_filter:
            print(f"按任务过滤: {self.tasks_filter}")
            filtered_episodes = []
            for task_idx, info in task_info.items():
                if info["name"] in self.tasks_filter:
                    filtered_episodes.extend(info["episodes"])
            available_episodes = list(set(filtered_episodes))
        else:
            # 使用所有episodes
            available_episodes = list(
                self.dataset_analysis["episode_info"].keys())

        # 限制episodes数量
        max_episodes = self.episode_end - self.episode_start
        if len(available_episodes) <= max_episodes:
            selected_episodes = available_episodes
            print(
                f"可用episodes数 ({len(available_episodes)}) <= max_episodes ({max_episodes})，使用所有可用episodes"
            )
        else:
            # 尝试从每个任务中均匀选择episodes
            episodes_per_task = max(1, max_episodes // len(task_info))
            selected_episodes = []

            for task_idx, info in task_info.items():
                if info["episodes"]:
                    # 从该任务的episodes中选择
                    task_eps = [
                        ep for ep in info["episodes"] if ep in available_episodes
                    ]
                    selected_from_task = task_eps[:episodes_per_task]
                    selected_episodes.extend(selected_from_task)

            # 如果还没达到max_episodes，随机选择剩余的
            if len(selected_episodes) < max_episodes:
                remaining_episodes = [
                    ep for ep in available_episodes if ep not in selected_episodes
                ]
                additional_needed = max_episodes - len(selected_episodes)
                additional_episodes = remaining_episodes[:additional_needed]
                selected_episodes.extend(additional_episodes)

            # 确保不超过max_episodes
            selected_episodes = selected_episodes[:max_episodes]

        selected_episodes.sort()
        print(
            f"选择了 {len(selected_episodes)} 个episodes进行评测: {selected_episodes[:10]}{'...' if len(selected_episodes) > 10 else ''}"
        )

        # 按任务打印选择的episode分布
        task_episode_counts = defaultdict(int)
        for ep_idx in selected_episodes:
            ep_tasks = self.dataset_analysis["episode_info"][ep_idx]["tasks"]
            for task_str in ep_tasks:
                task_idx = None
                for tid, tname in task_info.items():
                    if tname["name"] == task_str:
                        task_idx = tid
                        break
                if task_idx is not None:
                    task_episode_counts[task_idx] += 1

        print("选择的episodes按任务分布:")
        for task_idx, count in task_episode_counts.items():
            task_display_name = task_info[task_idx]["display_name"]
            print(f"  任务 {task_idx} ({task_display_name}): {count} episodes")

        return selected_episodes

    def evaluate_episode(self, episode_idx: int) -> Dict[str, Any]:
        """使用policy评测单个episode (TensorFlow RLDS格式，兼容droid_100_rlds features.json结构)"""
        # print(f"评测Episode {episode_idx}")
        import numpy as np
        from openpi_client import image_tools

        def to_numpy_safe(t):
            if hasattr(t, "numpy"):
                t = t.numpy()
            if isinstance(t, bytes):
                t = np.frombuffer(t, dtype=np.uint8)
            return t

        def to_str_safe(t):
            if hasattr(t, "numpy"):
                t = t.numpy()
            if isinstance(t, bytes):
                t = t.decode("utf-8")
            if not isinstance(t, str):
                t = str(t)
            return t

        # 遍历TF数据集，找到对应episode
        # i=0
        for ep_num, episode in enumerate(self.dataset):
            if ep_num == episode_idx:
                steps = episode["steps"]
                # 将steps数据集转换为列表以支持索引访问
                steps_list = list(steps)
                pred_actions = []
                errors = []
                gt_actions = []
                for i in range(0, len(steps_list)-self.args.chunk_size, self.args.chunk_size):
                    obs = steps_list[i]["observation"]
                    # 图像处理
                    ext_img = image_tools.resize_with_pad(
                        to_numpy_safe(obs["exterior_image_1_left"]), 224, 224
                    )
                    wrist_img = image_tools.resize_with_pad(
                        to_numpy_safe(obs["wrist_image_left"]), 224, 224
                    )
                    cartesian_pos = obs["cartesian_position"]
                    gripper_pos = obs["gripper_position"]
                    # print(f"gripper_pos: {gripper_pos}")
                    prompt = steps_list[i].get("language_instruction", "")
                    prompt = to_str_safe(prompt)

                    request_data = {
                        "observation/exterior_image_1_left": ext_img,
                        "observation/wrist_image_left": wrist_img,
                        "observation/cartesian_position": cartesian_pos,
                        "observation/gripper_position": gripper_pos,
                        "prompt": prompt,
                    }
                    # policy推理
                    pred_action = self.policy.infer(
                        request_data)["actions"][:self.args.chunk_size]
                    print(f"gripper_pos_pred: {pred_action[:,-1]}")

                    # if i == 0:
                    #     print(f"pred_action: {pred_action.shape}")
                    # Binarize gripper action
                    # 将gripper_position 0 1化 目前范围为0到100 归一化
                    pred_action[:, -1] = pred_action[:, -1] * 100
                    # pred_action = np.clip(pred_action, -1, 1)

                    # print(f"pred_action: {pred_action.shape}")

                    for j in range(self.args.chunk_size):
                        grip = 1.0 if pred_action[j, -1] > 0.5 else 0.0
                        action_j = np.concatenate(
                            [pred_action[j, :-1], [grip]])  # 新变量
                        pred_actions.append(action_j)
                        gt_action = np.concatenate(
                            [
                                steps_list[i +
                                           j]["action_dict"]["cartesian_position"],
                                steps_list[i +
                                           j]["action_dict"]["gripper_position"],
                            ]
                        )
                        gt_actions.append(gt_action)
                        error = np.mean(
                            (np.array(pred_action[j]) - np.array(gt_action)) ** 2)
                        errors.append(error)
                avg_mse = float(np.mean(errors))
                print(f"Episode {episode_idx} 平均MSE: {avg_mse:.4f}")
                return {
                    "episode_idx": episode_idx,
                    "num_frames": len(steps_list),
                    "avg_mse": avg_mse,
                    "pred_actions": np.array(pred_actions).tolist(),
                    "gt_actions": np.array(gt_actions).tolist(),
                }
        print(f"未找到Episode {episode_idx}")
        return {}

    def load_dataset(self) -> None:
        """使用tfds.load方式加载DROID RLDS数据集"""
        import tensorflow_datasets as tfds

        dataset_name = self.args.dataset_name
        data_dir = str(self.args.dataset_path)
        print(f"使用tfds.load加载RLDS数据集: {dataset_name}, data_dir={data_dir}")
        self.dataset = tfds.load(
            dataset_name,
            data_dir=data_dir,
            split="train",
        )
        print(f"已加载数据集: {dataset_name}")

    def load_policy(self) -> None:
        """Load OpenPI policy using config and checkpoint."""
        print(f"Loading OpenPI policy with config: {self.args.config}")
        print(f"Checkpoint directory: {self.args.checkpoint_dir}")

        # Get config
        config = _config.get_config(self.args.config)

        # Create trained policy
        self.policy = _policy_config.create_trained_policy(
            config, self.args.checkpoint_dir, default_prompt=self.args.default_prompt
        )

        print("OpenPI policy loaded successfully")

    def generate_plots(self, results: List[Dict]) -> None:
        """Generate evaluation plots."""
        if not results or not self.args.save_plots:
            return

        print("Generating evaluation plots...")

        # Extract metrics
        episodes = [r["episode_idx"] for r in results if "episode_idx" in r]
        avg_mse = [r.get("avg_mse", 0) for r in results]

        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # MSE over episodes
        if episodes and avg_mse:
            axes[0].plot(episodes, avg_mse, "b-o", label="MSE", alpha=0.7)
            axes[0].set_xlabel("Episode")
            axes[0].set_ylabel("MSE")
            axes[0].set_title("MSE by Episode")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

        # MSE distribution
        if avg_mse:
            axes[1].hist(
                avg_mse,
                bins=min(10, len(avg_mse)),
                alpha=0.7,
                color="blue",
                label="MSE",
            )
            axes[1].set_xlabel("MSE")
            axes[1].set_ylabel("Frequency")
            axes[1].set_title("MSE Distribution")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        # Action prediction vs ground truth (for the first episode)
        if results and "pred_actions" in results[0] and "gt_actions" in results[0]:
            import numpy as np

            pred = np.array(results[0]["pred_actions"])
            gt = np.array(results[0]["gt_actions"])
            num_steps, action_dim = pred.shape
            for d in range(action_dim):
                axes[2].plot(pred[:, d], label=f"Pred d{d}", linestyle="-")
                axes[2].plot(gt[:, d], label=f"GT d{d}", linestyle=":")
            axes[2].set_xlabel("Step")
            axes[2].set_ylabel("Action Value")
            axes[2].set_title("Pred vs GT Actions (Episode 0)")
            axes[2].legend(fontsize=8, ncol=2)
            axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = (
            self.plots_dir
            / f"evaluation_summary_{self.args.episode_start}_{self.args.episode_end}.png"
        )
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Plots saved to {plot_path}")

        # 为每个episode生成actions_joint_gripper_compare图
        print("Generating per-episode action comparison plots...")
        for result in results:
            if "pred_actions" in result and "gt_actions" in result:
                episode_idx = result["episode_idx"]
                pred = np.array(result["pred_actions"])
                gt = np.array(result["gt_actions"])

                # 创建每个episode的关节对比图
                fig_joint, axes_joint = plt.subplots(
                    7, 1, figsize=(10, 18), sharex=True
                )
                joint_names = ["x", "y", "z", "rx", "ry", "rz", "gripper"]

                for d in range(7):
                    axes_joint[d].plot(
                        pred[:, d], label="Pred", linestyle="-", linewidth=1.5
                    )
                    axes_joint[d].plot(
                        gt[:, d], label="GT", linestyle=":", linewidth=1.5
                    )
                    axes_joint[d].set_ylabel("Value")
                    axes_joint[d].set_title(
                        f"{joint_names[d]} - Episode {episode_idx}")
                    axes_joint[d].legend()
                    axes_joint[d].grid(True, alpha=0.3)

                axes_joint[-1].set_xlabel("Step")
                plt.tight_layout()

                # 保存图片，文件名包含episode索引
                plot_path_joint = (
                    self.plots_dir
                    / f"actions_joint_gripper_compare_ep{episode_idx}.png"
                )
                plt.savefig(plot_path_joint, dpi=300, bbox_inches="tight")
                plt.close(fig_joint)
                print(
                    f"Episode {episode_idx} action comparison plot saved to {plot_path_joint}"
                )

    def save_results(self, results: List[Dict]) -> None:
        """Save evaluation results."""
        # Save detailed results
        detailed_path = (
            self.output_dir
            / f"detailed_results_{self.args.episode_start}_{self.args.episode_end}.json"
        )
        with open(detailed_path, "w") as f:
            json.dump(results, f, indent=2)

        # Save metrics summary
        metrics_summary = self.metrics.get_summary()
        summary_path = (
            self.output_dir
            / f"metrics_summary_{self.args.episode_start}_{self.args.episode_end}.json"
        )
        with open(summary_path, "w") as f:
            json.dump(metrics_summary, f, indent=2)

        # Save CSV summary
        if results:
            csv_data = []
            for result in results:
                row = {
                    "episode_idx": result.get("episode_idx", -1),
                    "num_frames": result.get("num_frames", 0),
                    "avg_mse": result.get("avg_mse", 0),
                }
                csv_data.append(row)

            df = pd.DataFrame(csv_data)
            csv_path = (
                self.output_dir
                / f"results_summary_{self.args.episode_start}_{self.args.episode_end}.csv"
            )
            df.to_csv(csv_path, index=False)
            print(f"CSV summary saved to {csv_path}")

        print(f"Results saved to {self.output_dir}")

    def run_evaluation(self) -> Dict[str, Any]:
        """完整运行DROID数据集评测流程。"""
        print("开始DROID数据集评测流程\n")
        # 1. 加载数据集和policy

        self.load_policy()

        time.sleep(10)

        self.load_dataset()

        # time.sleep(10)
        # self.load_policy()

        # 2. 分析数据集任务分布
        # print("\n分析数据集任务分布...")
        # self.dataset_analysis = self.analyze_dataset_tasks()

        # 3. 选择要评测的episodes
        # print("\n选择评测的episodes...")
        # selected_episodes = self.select_episodes_for_evaluation()
        # print(f"最终将评测 {len(selected_episodes)} 个episodes\n")

        # 最大的episode
        max_episode = len(self.dataset)

        # 4. 遍历评测并收集结果

        # 验证episode范围
        if self.args.episode_start >= self.args.episode_end:
            raise ValueError("episode_start 必须小于 episode_end")
        elif self.args.episode_start < 0:
            print(f"episode_start {self.args.episode_start} 小于0,设置为0")
            self.args.episode_start = 0
        elif self.args.episode_end > max_episode:
            print(
                f"episode_end {self.args.episode_end} 大于最大episode {max_episode},设置为最大episode"
            )
            self.args.episode_end = max_episode

        selected_episodes = [
            i for i in range(self.args.episode_start, self.args.episode_end)
        ]

        results = []
        from tqdm import tqdm

        for ep_idx in tqdm(selected_episodes, desc="评测进度"):
            try:
                ep_result = self.evaluate_episode(ep_idx)
                if ep_result:
                    results.append(ep_result)
                    self.metrics.add_episode_metrics(ep_result)
            except Exception as e:
                print(f"评测Episode {ep_idx}时发生异常: {e}")
                import traceback

                traceback.print_exc()

        # 5. 保存评测结果
        print("\n保存评测结果...")
        self.save_results(results)

        # 6. 生成可视化图表
        print("\n生成评测可视化图表...")
        self.generate_plots(results)

        print("\nDROID数据集评测流程完成！")
        return {
            "results": results,
            "metrics_summary": self.metrics.get_summary(),
        }


def main():
    """DROID数据集评测主入口，支持命令行参数和异常捕获。"""
    print("启动数据集评测脚本\n")
    args = tyro.cli(Args)
    args.checkpoint_dir = _require_checkpoint_dir(args.checkpoint_dir)
    print(f"评测参数: {args}\n")
    try:
        # 创建评测器
        evaluator = DROIDDatasetEvaluator(args)
        # 运行评测
        evaluator.run_evaluation()
    except Exception as e:
        print(f"评测过程中发生异常: {e}")
        import traceback

        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
