#!/usr/bin/env python3
"""
π0.5 + RTC LIBERO Evaluation Script

Implements comprehensive evaluation metrics from:
"Real-Time Execution of Action Chunking Flow Policies"

Metrics implemented:
1. Task Success - Success rate on LIBERO benchmarks
2. Temporal Smoothness:
   - Boundary (Seam) Consistency (Bjump, Bratio)
   - Frequency Regularity (HF ratio)
   - Global Variation (TV-L1)
3. Ablation Study - Compare different overlap sizes and blending methods
"""

import sys
import os

# Add libero path
libero_path = os.path.join(os.path.dirname(__file__), "../third_party", "libero")
if libero_path not in sys.path:
    sys.path.insert(0, libero_path)

import collections
import dataclasses
import logging
import math
import pathlib
import time
import json
from typing import Any, Dict, List, Tuple, Optional
import statistics

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path, set_libero_default_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
import tqdm
import tyro
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from openpi.policies.rtc_policy import RTCPolicy, RTCConfig, RTCPolicyWrapper
from openpi.shared import metrics as _metrics

# Fix PyTorch 2.6+ torch.load weights_only issue for LIBERO init_states
# Add all common numpy types to safe globals
safe_numpy_types = [
    np.core.multiarray._reconstruct,
    np.ndarray,
    np.dtype,
]

# Add all numpy dtypes
try:
    import numpy.dtypes as np_dtypes
    for attr_name in dir(np_dtypes):
        attr = getattr(np_dtypes, attr_name)
        if isinstance(attr, type):
            safe_numpy_types.append(attr)
except Exception:
    pass

# Add legacy numpy scalar types
try:
    for dtype_name in ['float64', 'float32', 'int64', 'int32', 'bool_', 'uint8']:
        if hasattr(np, dtype_name):
            safe_numpy_types.append(getattr(np, dtype_name))
except Exception:
    pass

torch.serialization.add_safe_globals(safe_numpy_types)

# Monkey-patch torch.load to use weights_only=False for LIBERO compatibility
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

# Set matplotlib fonts
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial Unicode MS", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


@dataclasses.dataclass
class Args:
    """Evaluation arguments."""

    #################################################################################################################
    # Policy parameters
    #################################################################################################################
    config: str = "pi05_libero"  # Training config name
    checkpoint_dir: str = ""  # Required checkpoint directory
    default_prompt: str | None = None  # Default prompt

    #################################################################################################################
    # RTC parameters
    #################################################################################################################
    overlap_size: int = 4  # Overlap size O
    blending_method: str = 'linear'  # Blending method: 'linear', 'cosine', 'none'
    replan_interval: int = 5  # Replan every N steps

    #################################################################################################################
    # LIBERO environment parameters
    #################################################################################################################
    task_suite_name: str = "libero_object"  # Task suite
    num_steps_wait: int = 10  # Wait steps for objects to stabilize
    num_trials_per_task: int = 5  # Trials per task

    #################################################################################################################
    # Evaluation parameters
    #################################################################################################################
    output_dir: str = "outputs/libero/pi05_rtc_eval"  # Output directory
    seed: int = 42  # Random seed
    save_videos: bool = True  # Save videos
    save_pred_actions: bool = True  # Save per-episode predicted actions (NPZ)
    save_plots: bool = True  # Save plots
    detailed_logging: bool = True  # Detailed logging
    fc_ratio: float = 0.25  # High-frequency cutoff ratio normalized to Nyquist (0..1)

    # Ablation study
    run_ablation: bool = False  # Run ablation study
    ablation_overlaps: List[int] = dataclasses.field(
        default_factory=lambda: [0, 2, 4, 8]
    )  # Overlap sizes for ablation
    ablation_blendings: List[str] = dataclasses.field(
        default_factory=lambda: ['none', 'linear', 'cosine']
    )  # Blending methods for ablation


class TemporalMetrics:
    """Compute temporal smoothness metrics as defined in the paper."""

    @staticmethod
    def compute_boundary_consistency(
        chunk_transitions: List[Dict],
    ) -> Dict[str, float]:
        """
        Compute boundary (seam) consistency metrics.

        bjump: Average L2 distance at chunk boundaries
        bratio: Ratio of boundary jumps to average intra-chunk step magnitude

        Args:
            chunk_transitions: List of transition info dicts

        Returns:
            Dict with bjump and bratio
        """
        if len(chunk_transitions) == 0:
            return {'bjump': 0.0, 'bratio': 0.0}

        # Compute aligned overlap metrics with the canonical paper definition.
        boundary_jumps = []
        boundary_ratios = []
        for trans in chunk_transitions:
            prev_tail = np.asarray(trans.get('prev_tail', []))
            new_head = np.asarray(trans.get('new_head', []))
            if prev_tail.size == 0 or new_head.size == 0:
                continue
            seam = _metrics.compute_seam_metrics(prev_tail, new_head)
            boundary_jumps.append(seam['bjump'])
            boundary_ratios.append(seam['bratio'])

        bjump = float(np.mean(boundary_jumps)) if boundary_jumps else 0.0
        bratio = float(np.mean(boundary_ratios)) if boundary_ratios else 0.0

        return {
            'bjump': float(bjump),
            'bratio': float(bratio),
            'boundary_jumps': [float(j) for j in boundary_jumps]
        }

    @staticmethod
    def compute_frequency_regularity(action_history: np.ndarray, fc_ratio: float = 0.25) -> Dict[str, float]:
        """
        Compute frequency regularity metrics.

        hf_ratio: Ratio of high-frequency energy to total energy
        Lower is better (smoother actions)

        Args:
            action_history: Full action sequence [T, A]
            fc_ratio: Cutoff ratio normalized to Nyquist in [0, 1]

        Returns:
            Dict with hf_ratio and related metrics
        """
        return {'hf_ratio': _metrics.compute_hf_ratio(action_history, cutoff_ratio=fc_ratio)}

    @staticmethod
    def compute_global_variation(action_history: np.ndarray) -> Dict[str, float]:
        """
        Compute global variation metrics.

        tv_l1: Mean per-step L1 norm of first differences
        msd_d1: Mean squared displacement of first differences
        msd_d2: Mean squared displacement of second differences
        msd_d3: Mean squared displacement of third differences

        Args:
            action_history: Full action sequence [T, A]

        Returns:
            Dict with variation metrics
        """
        return {
            'tv_l1': _metrics.compute_tv_l1(action_history),
            'msd_d1': _metrics.compute_msd_delta(action_history, order=1),
            'msd_d2': _metrics.compute_msd_delta(action_history, order=2),
            'msd_d3': _metrics.compute_msd_delta(action_history, order=3),
        }


def _require_checkpoint_dir(checkpoint_dir: str) -> str:
    """Reject an omitted checkpoint before environment or model initialization."""
    if not checkpoint_dir.strip():
        raise ValueError("--checkpoint-dir is required")
    return checkpoint_dir


class Pi05RTCEvaluator:
    """π0.5 + RTC evaluator for LIBERO."""

    def __init__(self, args: Args):
        self.args = args
        self.policy = None
        self.results = []
        self._setup_output_dirs()
        self._setup_policy()
        self._setup_environment()

    def _setup_output_dirs(self):
        """Setup output directories."""
        self.output_dir = pathlib.Path(self.args.output_dir)
        self.video_dir = self.output_dir / "videos"
        self.plots_dir = self.output_dir / "plots"
        self.data_dir = self.output_dir / "data"
        self.pred_actions_dir = self.data_dir / "pred_actions"

        for dir_path in [self.output_dir, self.video_dir, self.plots_dir, self.data_dir, self.pred_actions_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _setup_policy(self):
        """Setup RTC policy."""
        logging.info(f"Loading π0.5 policy: {self.args.config}")
        logging.info(f"Checkpoint: {self.args.checkpoint_dir}")
        logging.info(f"RTC config: overlap={self.args.overlap_size}, "
                    f"blending={self.args.blending_method}, "
                    f"replan_interval={self.args.replan_interval}")

        try:
            rtc_config = RTCConfig(
                overlap_size=self.args.overlap_size,
                blending_method=self.args.blending_method,
                replan_interval=self.args.replan_interval,
                track_metrics=True,
            )

            self.policy = RTCPolicyWrapper.create_from_checkpoint(
                config_name=self.args.config,
                checkpoint_dir=self.args.checkpoint_dir,
                rtc_config=rtc_config,
                default_prompt=self.args.default_prompt,
            )

            logging.info("π0.5 + RTC policy loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load policy: {e}")
            raise

    def _setup_environment(self):
        """Setup LIBERO environment."""
        np.random.seed(self.args.seed)

        benchmark_dict = benchmark.get_benchmark_dict()
        self.task_suite = benchmark_dict[self.args.task_suite_name]()
        self.num_tasks_in_suite = self.task_suite.n_tasks

        logging.info(f"Task suite: {self.args.task_suite_name}")
        logging.info(f"Number of tasks: {self.num_tasks_in_suite}")

        # Max steps per suite
        max_steps_dict = {
            "libero_spatial": 220,
            "libero_object": 280,
            "libero_goal": 300,
            "libero_10": 520,
            "libero_90": 400,
        }

        if self.args.task_suite_name not in max_steps_dict:
            raise ValueError(f"Unknown task suite: {self.args.task_suite_name}")

        self.max_steps = max_steps_dict[self.args.task_suite_name]
        logging.info(f"Max steps: {self.max_steps}")

    def _get_libero_env(self, task):
        """Initialize LIBERO environment."""
        task_description = task.language
        task_bddl_file = (
            pathlib.Path(get_libero_path("bddl_files"))
            / task.problem_folder
            / task.bddl_file
        )
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": LIBERO_ENV_RESOLUTION,
            "camera_widths": LIBERO_ENV_RESOLUTION,
        }
        env = OffScreenRenderEnv(**env_args)
        env.seed(self.args.seed)
        return env, task_description

    def _quat2axisangle(self, quat):
        """Quaternion to axis-angle."""
        if quat[3] > 1.0:
            quat[3] = 1.0
        elif quat[3] < -1.0:
            quat[3] = -1.0

        den = np.sqrt(1.0 - quat[3] * quat[3])
        if math.isclose(den, 0.0):
            return np.zeros(3)

        return (quat[:3] * 2.0 * math.acos(quat[3])) / den

    def _prepare_observation(self, obs, task_description):
        """Prepare observation for policy."""
        # Get images (rotate 180 degrees to match training)
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

        # Resize
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        )
        wrist_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, 224, 224)
        )

        # Prepare observation dict
        element = {
            "observation/image": img,
            "observation/wrist_image": wrist_img,
            "observation/state": np.concatenate((
                obs["robot0_eef_pos"],
                self._quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )),
            "prompt": str(task_description),
        }

        return element, img, wrist_img

    def _run_episode(
        self,
        task_id: int,
        episode_idx: int,
        task,
        task_description: str,
        initial_states
    ) -> Tuple[bool, Dict[str, Any]]:
        """Run single episode with RTC policy."""
        # Initialize environment
        env, _ = self._get_libero_env(task)
        env.reset()

        # Reset RTC policy state
        self.policy.reset()

        # Set initial state
        obs = env.set_init_state(initial_states[episode_idx])

        # Episode tracking
        t = 0
        replay_images = []
        wrist_images = []
        actions_taken = []
        success = False

        episode_start_time = time.time()
        inference_times = []

        # ARL tracking: track chunk inference times and action contributions
        chunk_inference_times = []  # T_c for each chunk
        chunk_action_counts = []    # |a_c| for each chunk
        chunk_count = 0
        actions_from_current_chunk = 0

        logging.info(f"Starting episode {episode_idx + 1}...")

        while t < self.max_steps + self.args.num_steps_wait:
            try:
                # Wait for objects to stabilize
                if t < self.args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # Prepare observation
                element, img, wrist_img = self._prepare_observation(obs, task_description)

                # Save images
                if self.args.save_videos:
                    replay_images.append(img)
                    wrist_images.append(wrist_img)

                # Get action from RTC policy
                inference_start = time.time()
                result = self.policy.infer(element)
                action = result['actions']
                inference_time = (time.time() - inference_start) * 1000
                inference_times.append(inference_time)

                # Track for ARL: detect when a new chunk is generated.
                # RTC policy replans every replan_interval steps
                if (len(actions_taken) % self.args.replan_interval) == 0:
                    if chunk_count > 0:
                        chunk_action_counts.append(actions_from_current_chunk)
                    chunk_inference_times.append(inference_time)
                    chunk_count += 1
                    actions_from_current_chunk = 0

                # Execute action
                obs, reward, done, info = env.step(action.tolist())
                actions_taken.append(action.copy())
                actions_from_current_chunk += 1

                if done:
                    success = True
                    break

                t += 1

            except Exception as e:
                logging.error(f"Episode execution error: {e}")
                break

        episode_duration = time.time() - episode_start_time

        # Finalize ARL tracking: add last chunk's contribution
        if chunk_count > 0:
            chunk_action_counts.append(actions_from_current_chunk)

        # ARL is total chunk inference time divided by total executed actions.
        arl = 0.0
        if (
            len(chunk_inference_times) > 0
            and len(chunk_action_counts) > 0
            and sum(chunk_action_counts) > 0
        ):
            arl = _metrics.compute_average_reasoning_latency(
                chunk_inference_times,
                chunk_action_counts,
            )

        # Get RTC metrics
        rtc_metrics = self.policy.get_metrics()

        # Compute temporal metrics
        if len(actions_taken) > 0:
            action_history = np.array(actions_taken)

            # Boundary consistency
            boundary_metrics = TemporalMetrics.compute_boundary_consistency(
                rtc_metrics.get('chunk_transitions', [])
            )

            # Frequency regularity
            freq_metrics = TemporalMetrics.compute_frequency_regularity(action_history, fc_ratio=self.args.fc_ratio)

            # Global variation
            var_metrics = TemporalMetrics.compute_global_variation(action_history)

            temporal_metrics = {
                **boundary_metrics,
                **freq_metrics,
                **var_metrics,
            }
        else:
            temporal_metrics = {}

        # Compile episode data
        episode_data = {
            'task_id': task_id,
            'episode_idx': episode_idx,
            'task_description': task_description,
            'success': success,
            'steps': t,
            'duration': episode_duration,
            'avg_inference_time': statistics.mean(inference_times) if inference_times else 0,
            'inference_times': inference_times,
            'arl': arl,  # Average Reasoning Latency (ms)
            'chunk_inference_times': chunk_inference_times,
            'chunk_action_counts': chunk_action_counts,
            'num_chunks': len(chunk_inference_times),
            'temporal_metrics': temporal_metrics,
            'rtc_config': {
                'overlap_size': self.args.overlap_size,
                'blending_method': self.args.blending_method,
                'replan_interval': self.args.replan_interval,
            }
        }

        # Save episode data
        episode_data_path = self.data_dir / f"episode_{task_id}_{episode_idx}.json"
        with open(episode_data_path, 'w') as f:
            json.dump(episode_data, f, indent=2)

        # Save predicted actions for external smoothness eval compatibility
        if self.args.save_pred_actions and len(actions_taken) > 0:
            pred_arr = np.asarray(actions_taken, dtype=np.float32)
            npz_path = self.pred_actions_dir / f"task{task_id:02d}_ep{episode_idx:02d}.npz"
            np.savez_compressed(
                npz_path.as_posix(),
                pred_actions=pred_arr,
                task_id=int(task_id),
                episode_idx=int(episode_idx),
                success=bool(success),
                task_description=str(task_description),
            )

        # Save video
        if self.args.save_videos and replay_images:
            self._save_video(
                task_id, episode_idx, task_description, success,
                replay_images, wrist_images, actions_taken
            )

        env.close()
        return success, episode_data

    def _save_video(
        self,
        task_id: int,
        episode_idx: int,
        task_description: str,
        success: bool,
        replay_images: List,
        wrist_images: List,
        actions: List
    ):
        """Save episode video."""
        suffix = "success" if success else "failure"
        task_segment = task_description.replace(" ", "_")
        video_path = self.video_dir / f"rtc_{task_segment}_{episode_idx}_{suffix}.mp4"

        # Create side-by-side video
        frames = []
        for i in range(min(len(replay_images), len(wrist_images))):
            frame = np.hstack([replay_images[i], wrist_images[i]])
            frames.append(frame)

        if frames:
            imageio.mimwrite(video_path, frames, fps=10)
            logging.info(f"Video saved: {video_path}")

    def evaluate(self) -> Dict[str, Any]:
        """Run full evaluation."""
        logging.info("Starting π0.5 + RTC evaluation...")

        total_episodes = 0
        total_successes = 0
        all_results = []

        for task_id in tqdm.tqdm(range(self.num_tasks_in_suite), desc="Evaluating tasks"):
            task = self.task_suite.get_task(task_id)
            initial_states = self.task_suite.get_task_init_states(task_id)
            _, task_description = self._get_libero_env(task)

            logging.info(f"\nTask {task_id + 1}/{self.num_tasks_in_suite}: {task_description}")

            task_successes = 0

            for episode_idx in range(self.args.num_trials_per_task):
                success, episode_data = self._run_episode(
                    task_id, episode_idx, task, task_description, initial_states
                )

                all_results.append(episode_data)

                if success:
                    task_successes += 1
                    total_successes += 1

                total_episodes += 1

                if self.args.detailed_logging:
                    logging.info(f"Episode {episode_idx + 1}: {'Success' if success else 'Failure'}")
                    logging.info(f"  ARL: {episode_data.get('arl', 0):.2f} ms")
                    if 'temporal_metrics' in episode_data:
                        tm = episode_data['temporal_metrics']
                        logging.info(f"  bjump: {tm.get('bjump', 0):.4f}")
                        logging.info(f"  hf_ratio: {tm.get('hf_ratio', 0):.4f}")
                        logging.info(f"  tv_l1: {tm.get('tv_l1', 0):.4f}")

            task_success_rate = task_successes / self.args.num_trials_per_task
            logging.info(f"Task {task_id + 1} success rate: {task_success_rate:.3f}")

        # Generate final report
        final_results = self._generate_final_report(total_episodes, total_successes, all_results)

        # Create plots
        if self.args.save_plots:
            self._create_plots(all_results)

        return final_results

    def _generate_final_report(
        self,
        total_episodes: int,
        total_successes: int,
        all_results: List[Dict]
    ) -> Dict[str, Any]:
        """Generate final evaluation report."""
        overall_success_rate = total_successes / total_episodes if total_episodes > 0 else 0

        # Aggregate ARL (Average Reasoning Latency)
        arl_values = [r.get('arl', 0) for r in all_results if r.get('arl', 0) > 0]
        arl_stats = {}
        if arl_values:
            arl_stats = {
                'mean': float(np.mean(arl_values)),
                'std': float(np.std(arl_values)),
                'min': float(np.min(arl_values)),
                'max': float(np.max(arl_values)),
                'p50': float(np.percentile(arl_values, 50)),
                'p90': float(np.percentile(arl_values, 90)),
            }

        # Aggregate temporal metrics
        temporal_metrics_agg = {}
        metric_keys = ['bjump', 'bratio', 'hf_ratio', 'tv_l1', 'msd_d1', 'msd_d2', 'msd_d3']

        for key in metric_keys:
            values = [
                r['temporal_metrics'].get(key, 0)
                for r in all_results
                if 'temporal_metrics' in r
            ]
            if values:
                temporal_metrics_agg[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                }

        # Compile report
        report = {
            'evaluation_summary': {
                'task_suite': self.args.task_suite_name,
                'total_episodes': total_episodes,
                'total_successes': total_successes,
                'overall_success_rate': overall_success_rate,
                'evaluation_time': time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            'rtc_config': {
                'overlap_size': self.args.overlap_size,
                'blending_method': self.args.blending_method,
                'replan_interval': self.args.replan_interval,
            },
            'arl': arl_stats,  # Average Reasoning Latency statistics
            'temporal_metrics': temporal_metrics_agg,
            'episodes': all_results,
        }

        # Save report
        report_path = self.output_dir / "evaluation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Print summary
        logging.info("\n" + "=" * 80)
        logging.info("π0.5 + RTC Evaluation Report")
        logging.info("=" * 80)
        logging.info(f"Task suite: {self.args.task_suite_name}")
        logging.info(f"Overall success rate: {overall_success_rate:.3f}")
        logging.info(f"Total episodes: {total_episodes}")
        logging.info(f"Total successes: {total_successes}")
        logging.info(f"\nRTC Configuration:")
        logging.info(f"  Overlap size (O): {self.args.overlap_size}")
        logging.info(f"  Blending method: {self.args.blending_method}")
        logging.info(f"  Replan interval: {self.args.replan_interval}")

        if arl_stats:
            logging.info(f"\nAverage Reasoning Latency (ARL):")
            logging.info(f"  Mean: {arl_stats['mean']:.2f} ms")
            logging.info(f"  Std: {arl_stats['std']:.2f} ms")
            logging.info(f"  Median (p50): {arl_stats['p50']:.2f} ms")
            logging.info(f"  p90: {arl_stats['p90']:.2f} ms")

        logging.info(f"\nTemporal Smoothness Metrics:")
        for key, stats in temporal_metrics_agg.items():
            logging.info(f"  {key}: {stats['mean']:.4f} ± {stats['std']:.4f}")

        logging.info(f"\nFull report saved to: {report_path}")

        return report

    def _create_plots(self, all_results: List[Dict]):
        """Create visualization plots."""
        logging.info("Creating visualization plots...")

        # Extract metrics
        success_rates = []
        bjumps = []
        hf_ratios = []
        tv_l1s = []

        for r in all_results:
            success_rates.append(1 if r['success'] else 0)
            if 'temporal_metrics' in r:
                tm = r['temporal_metrics']
                bjumps.append(tm.get('bjump', 0))
                hf_ratios.append(tm.get('hf_ratio', 0))
                tv_l1s.append(tm.get('tv_l1', 0))

        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Success rate
        axes[0, 0].bar(['Success', 'Failure'],
                      [sum(success_rates), len(success_rates) - sum(success_rates)],
                      color=['green', 'red'], alpha=0.7)
        axes[0, 0].set_title('Task Success')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].grid(True, alpha=0.3)

        # Boundary consistency
        if bjumps:
            axes[0, 1].hist(bjumps, bins=20, alpha=0.7, color='skyblue')
            axes[0, 1].set_title(f'Boundary Consistency (bjump)\nMean: {np.mean(bjumps):.4f}')
            axes[0, 1].set_xlabel('bjump')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)

        # Frequency regularity
        if hf_ratios:
            axes[1, 0].hist(hf_ratios, bins=20, alpha=0.7, color='lightgreen')
            axes[1, 0].set_title(f'Frequency Regularity (hf_ratio)\nMean: {np.mean(hf_ratios):.4f}')
            axes[1, 0].set_xlabel('hf_ratio')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)

        # Global variation
        if tv_l1s:
            axes[1, 1].hist(tv_l1s, bins=20, alpha=0.7, color='coral')
            axes[1, 1].set_title(f'Global Variation (tv_l1)\nMean: {np.mean(tv_l1s):.4f}')
            axes[1, 1].set_xlabel('tv_l1')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'rtc_metrics_summary.png', dpi=200, bbox_inches='tight')
        plt.close()

        logging.info(f"Plots saved to: {self.plots_dir}")

    def run_ablation_study(self) -> Dict[str, Any]:
        """Run ablation study over different RTC configurations."""
        logging.info("Starting ablation study...")

        ablation_results = []

        for overlap in self.args.ablation_overlaps:
            for blending in self.args.ablation_blendings:
                logging.info(f"\n{'='*80}")
                logging.info(f"Ablation: overlap={overlap}, blending={blending}")
                logging.info(f"{'='*80}")

                # Update config
                self.args.overlap_size = overlap
                self.args.blending_method = blending

                # Recreate policy with new config
                self._setup_policy()

                # Run evaluation
                results = self.evaluate()

                ablation_results.append({
                    'overlap_size': overlap,
                    'blending_method': blending,
                    'results': results,
                })

        # Save ablation results
        ablation_path = self.output_dir / "ablation_study.json"
        with open(ablation_path, 'w') as f:
            json.dump(ablation_results, f, indent=2)

        # Create ablation plots
        self._create_ablation_plots(ablation_results)

        logging.info(f"\nAblation study complete. Results saved to: {ablation_path}")

        return ablation_results

    def _create_ablation_plots(self, ablation_results: List[Dict]):
        """Create ablation study plots."""
        logging.info("Creating ablation study plots...")

        # Extract data
        configs = []
        success_rates = []
        bjumps = []
        hf_ratios = []
        tv_l1s = []

        for result in ablation_results:
            config_name = f"O={result['overlap_size']}, {result['blending_method']}"
            configs.append(config_name)

            summary = result['results']['evaluation_summary']
            success_rates.append(summary['overall_success_rate'])

            tm = result['results']['temporal_metrics']
            bjumps.append(tm.get('bjump', {}).get('mean', 0))
            hf_ratios.append(tm.get('hf_ratio', {}).get('mean', 0))
            tv_l1s.append(tm.get('tv_l1', {}).get('mean', 0))

        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Success rate
        axes[0, 0].bar(range(len(configs)), success_rates, alpha=0.7, color='green')
        axes[0, 0].set_xticks(range(len(configs)))
        axes[0, 0].set_xticklabels(configs, rotation=45, ha='right')
        axes[0, 0].set_title('Success Rate by Configuration')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].grid(True, alpha=0.3)

        # Bjump
        axes[0, 1].bar(range(len(configs)), bjumps, alpha=0.7, color='skyblue')
        axes[0, 1].set_xticks(range(len(configs)))
        axes[0, 1].set_xticklabels(configs, rotation=45, ha='right')
        axes[0, 1].set_title('Boundary Consistency (bjump)')
        axes[0, 1].set_ylabel('bjump')
        axes[0, 1].grid(True, alpha=0.3)

        # HF ratio
        axes[1, 0].bar(range(len(configs)), hf_ratios, alpha=0.7, color='lightgreen')
        axes[1, 0].set_xticks(range(len(configs)))
        axes[1, 0].set_xticklabels(configs, rotation=45, ha='right')
        axes[1, 0].set_title('Frequency Regularity (hf_ratio)')
        axes[1, 0].set_ylabel('hf_ratio')
        axes[1, 0].grid(True, alpha=0.3)

        # TV-L1
        axes[1, 1].bar(range(len(configs)), tv_l1s, alpha=0.7, color='coral')
        axes[1, 1].set_xticks(range(len(configs)))
        axes[1, 1].set_xticklabels(configs, rotation=45, ha='right')
        axes[1, 1].set_title('Global Variation (tv_l1)')
        axes[1, 1].set_ylabel('tv_l1')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'ablation_study.png', dpi=200, bbox_inches='tight')
        plt.close()

        logging.info(f"Ablation plots saved to: {self.plots_dir}")


def main(args: Args) -> None:
    """Main evaluation function."""
    args.checkpoint_dir = _require_checkpoint_dir(args.checkpoint_dir)

    # Setup LIBERO paths
    try:
        correct_benchmark_root = os.path.abspath(
            os.path.join(libero_path, "libero", "libero")
        )
        set_libero_default_path(correct_benchmark_root)
        logging.info(f"LIBERO configured: {correct_benchmark_root}")
    except Exception as e:
        logging.warning(f"Failed to update LIBERO config: {e}")

    # Create evaluator
    evaluator = Pi05RTCEvaluator(args)

    # Run evaluation or ablation study
    if args.run_ablation:
        evaluator.run_ablation_study()
    else:
        evaluator.evaluate()

    logging.info("π0.5 + RTC evaluation complete!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )
    tyro.cli(main)
