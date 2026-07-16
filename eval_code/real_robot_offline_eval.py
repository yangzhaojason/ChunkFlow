#!/usr/bin/env python3
"""Offline real-robot RLDS evaluation for ChunkFlow checkpoints.

This script performs offline evaluation of OpenPI models on an RLDS dataset:
1. Loads DROID dataset using tfds format
2. Uses OpenPI policy loading mechanism
3. Performs action predictions with chunk processing
4. Saves predictions and ground truth data
5. Generates video visualizations

Usage:
    uv run eval_code/real_robot_offline_eval.py \
      --checkpoint-dir checkpoints/pi05_truth_cartesian_wood/run/49999 \
      --dataset-path datasets/real_robot \
      --dataset-name chunkflow_real_wood \
      --config pi05_truth_finetune_cartesian_wood \
      --episode-start 10 --episode-end 15
"""

import cv2
import tensorflow as tf
from typing import Any, Dict, List, Optional
from pathlib import Path
from collections import defaultdict
import statistics
from openpi_client import image_tools
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config
from openpi.policies import policy as _policy
from tqdm import tqdm
import tyro
import torch
import numpy as np
import matplotlib.pyplot as plt
import dataclasses
import json
import os

# Set environment variables to avoid video processing issues
os.environ["LEROBOT_SKIP_VIDEO"] = "1"
os.environ["DISABLE_TORCHCODEC"] = "1"

tf.config.set_visible_devices([], "GPU")

# Set matplotlib style
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial Unicode MS", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


@dataclasses.dataclass
class Args:
    """Arguments for the DROID dataset evaluation script."""

    # Dataset parameters
    dataset_path: str = "datasets/real_robot"
    dataset_name: str = "chunkflow_real_wood"
    episode_start: int = 0
    episode_end: int = 10
    chunk_size: int = 10

    # Model parameters
    config: str = "pi05_truth_finetune_cartesian_wood"
    checkpoint_dir: str = ""
    default_prompt: Optional[str] = None
    resize_size: int = 224

    # Evaluation parameters
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Output parameters
    output_dir: str = "outputs/real_robot/offline_eval"
    save_plots: bool = False
    save_videos: bool = False


def _require_checkpoint_dir(checkpoint_dir: str) -> str:
    """Reject an omitted checkpoint before model or dataset initialization."""
    if not checkpoint_dir.strip():
        raise ValueError("--checkpoint-dir is required")
    return checkpoint_dir


class DROIDDatasetEvaluator:
    """Comprehensive DROID dataset evaluator using OpenPI models."""

    def __init__(self, args: Args):
        self.args = args
        self.dataset = None
        self.policy = None
        self.episode_start = args.episode_start
        self.episode_end = args.episode_end

        # Setup output directories
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"Initialized DROID evaluator with output dir: {self.output_dir}")

    def evaluate_episode(self, episode_idx: int) -> Dict[str, Any]:
        """使用policy评测单个episode"""
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

        # 创建输出目录
        data_save_dir = os.path.join(self.args.output_dir, "data")
        os.makedirs(data_save_dir, exist_ok=True)
        video_save_dir = os.path.join(self.args.output_dir, "videos")
        os.makedirs(video_save_dir, exist_ok=True)

        # 获取指定episode
        episode = list(self.dataset)[episode_idx]
        steps = episode["steps"]
        steps_list = list(steps)

        pred_actions = []
        gt_actions = []

        # 初始化视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30
        frame_width, frame_height = 224, 224

        exterior_video_path = os.path.join(
            video_save_dir, f"exterior_episode_{episode_idx}.mp4")
        wrist_video_path = os.path.join(
            video_save_dir, f"wrist_episode_{episode_idx}.mp4")

        exterior_video_writer = cv2.VideoWriter(
            exterior_video_path, fourcc, fps, (frame_width, frame_height))
        wrist_video_writer = cv2.VideoWriter(
            wrist_video_path, fourcc, fps, (frame_width, frame_height))

        # 处理每个chunk
        for i in range(0, len(steps_list) - self.args.chunk_size, self.args.chunk_size):
            # 保存视频帧
            for j in range(self.args.chunk_size):
                ext_img = image_tools.resize_with_pad(
                    to_numpy_safe(
                        steps_list[i+j]["observation"]["exterior_image_1_left"]), 224, 224
                )
                wrist_img = image_tools.resize_with_pad(
                    to_numpy_safe(
                        steps_list[i+j]["observation"]["wrist_image_left"]), 224, 224
                )

                # 转换为BGR格式并写入视频
                if ext_img.shape[2] == 3:
                    ext_img_bgr = cv2.cvtColor(ext_img, cv2.COLOR_RGB2BGR)
                else:
                    ext_img_bgr = ext_img

                if wrist_img.shape[2] == 3:
                    wrist_img_bgr = cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR)
                else:
                    wrist_img_bgr = wrist_img

                exterior_video_writer.write(ext_img_bgr)
                wrist_video_writer.write(wrist_img_bgr)

            # 准备推理数据
            ext_img_origin = image_tools.resize_with_pad(
                to_numpy_safe(steps_list[i]["observation"]
                              ["exterior_image_1_left"]), 224, 224
            )
            wrist_img_origin = image_tools.resize_with_pad(
                to_numpy_safe(steps_list[i]["observation"]
                              ["wrist_image_left"]), 224, 224
            )

            cartesian_pos = steps_list[i]["observation"]["cartesian_position"]
            gripper_pos = steps_list[i]["observation"]["gripper_position"]
            prompt = to_str_safe(steps_list[i].get("language_instruction", ""))

            request_data = {
                "observation/exterior_image_1_left": ext_img_origin,
                "observation/wrist_image_left": wrist_img_origin,
                "observation/cartesian_position": cartesian_pos,
                "observation/gripper_position": gripper_pos,
                "prompt": prompt,
            }

            if i == 0:
                init_joint = steps_list[i]["observation"]["joint_position"]

            # Policy推理
            pred_action = self.policy.infer(
                request_data)["actions"][:self.args.chunk_size]
            pred_action[:, -1] = pred_action[:, -1] / 100

            # 处理每个chunk中的动作
            for j in range(self.args.chunk_size):
                # 二值化gripper动作
                if pred_action[j, -1] > 0.5:
                    pred_action_j = np.concatenate(
                        [pred_action[j, :-1], np.ones((1,))])
                else:
                    pred_action_j = np.concatenate(
                        [pred_action[j, :-1], np.zeros((1,))])

                pred_actions.append(pred_action_j)

                gt_action = np.concatenate([
                    steps_list[i+j]["action_dict"]["cartesian_position"],
                    steps_list[i+j]["action_dict"]["gripper_position"] / 100,
                ])
                gt_actions.append(gt_action)

        # 保存结果
        pred_actions = np.stack(pred_actions)
        gt_actions = np.stack(gt_actions)

        np.savez(
            os.path.join(data_save_dir, f"episode_{episode_idx}.npz"),
            pred_actions=pred_actions,
            gt_actions=gt_actions,
            init_joint=init_joint
        )

        # 释放视频写入器资源
        exterior_video_writer.release()
        wrist_video_writer.release()

        print(f"Episode {episode_idx} 处理完成:")
        print(f"  - Exterior视频: {exterior_video_path}")
        print(f"  - Wrist视频: {wrist_video_path}")

        return {
            "episode_idx": episode_idx,
            "pred_actions": pred_actions,
            "gt_actions": gt_actions,
            "num_frames": len(pred_actions)
        }

    def load_dataset(self) -> None:
        """使用tfds.load方式加载DROID RLDS数据集"""
        import tensorflow_datasets as tfds

        # dataset_name = "droid_100"
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

    def save_results(self, results: List[Dict]) -> None:
        """保存评测结果"""
        # 转换numpy数组为列表以便JSON序列化
        json_results = []
        for result in results:
            json_result = result.copy()
            if "pred_actions" in json_result:
                json_result["pred_actions"] = json_result["pred_actions"].tolist()
            if "gt_actions" in json_result:
                json_result["gt_actions"] = json_result["gt_actions"].tolist()
            json_results.append(json_result)

        # 保存详细结果
        detailed_path = self.output_dir / \
            f"detailed_results_{self.args.episode_start}_{self.args.episode_end}.json"
        with open(detailed_path, "w") as f:
            json.dump(json_results, f, indent=2)

        print(f"结果已保存到 {self.output_dir}")

    def run_evaluation(self) -> Dict[str, Any]:
        """运行DROID数据集评测流程"""
        print("开始DROID数据集评测流程\n")

        # 加载模型和数据集
        self.load_policy()
        self.load_dataset()

        # 验证episode范围
        max_episode = len(self.dataset)
        if self.args.episode_start >= self.args.episode_end:
            raise ValueError("episode_start 必须小于 episode_end")
        elif self.args.episode_start < 0:
            print(f"episode_start {self.args.episode_start} 小于0,设置为0")
            self.args.episode_start = 0
        elif self.args.episode_end > max_episode:
            print(
                f"episode_end {self.args.episode_end} 大于最大episode {max_episode},设置为最大episode")
            self.args.episode_end = max_episode

        selected_episodes = list(
            range(self.args.episode_start, self.args.episode_end))
        results = []

        # 评测每个episode
        for ep_idx in tqdm(selected_episodes, desc="评测进度"):
            try:
                ep_result = self.evaluate_episode(ep_idx)
                if ep_result:
                    results.append(ep_result)
            except Exception as e:
                print(f"评测Episode {ep_idx}时发生异常: {e}")
                import traceback
                traceback.print_exc()

        # 保存结果
        print("\n保存评测结果...")
        self.save_results(results)

        print("\nDROID数据集评测流程完成！")
        return {"results": results}


def main():
    """DROID数据集评测主入口，支持命令行参数和异常捕获。"""
    print("启动DROID数据集评测脚本\n")
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
