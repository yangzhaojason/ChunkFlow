"""
Real-Time Chunking (RTC) Policy Wrapper for π0.5

This module implements the RTC inference strategy from the paper:
"Real-Time Execution of Action Chunking Flow Policies"

Key features:
- Overlapping action chunks with configurable overlap O
- Temporal blending for smooth transitions
- Boundary consistency metrics
"""

import dataclasses
from typing import Any, Dict, Optional
import numpy as np
import logging
from collections import deque

logger = logging.getLogger("openpi")


@dataclasses.dataclass
class RTCConfig:
    """Configuration for Real-Time Chunking execution."""

    # Overlap size O: number of steps to overlap between chunks
    overlap_size: int = 4

    # Blending method: 'linear', 'cosine', or 'none'
    blending_method: str = 'linear'

    # Whether to use boundary consistency regularization during training
    use_boundary_loss: bool = False

    # Weight for boundary consistency loss
    boundary_loss_weight: float = 1.0

    # Replan interval: how often to request new chunks (in steps)
    replan_interval: int = 5

    # Whether to track temporal metrics
    track_metrics: bool = True


class RTCPolicy:
    """
    Real-Time Chunking Policy Wrapper.

    Wraps a base policy (e.g., π0.5) to provide RTC execution with:
    - Overlapping action chunks
    - Temporal blending for smooth transitions
    - Metric tracking for evaluation
    """

    def __init__(self, base_policy, config: RTCConfig):
        """
        Initialize RTC policy wrapper.

        Args:
            base_policy: Base policy that generates action chunks
            config: RTC configuration
        """
        self.base_policy = base_policy
        self.config = config

        # State for RTC execution
        self.current_chunk = None
        self.prev_chunk_tail = None
        self.step_in_chunk = 0
        self.total_steps = 0

        # Metrics tracking
        self.metrics = {
            'boundary_consistency': [],
            'chunk_transitions': [],
            'action_history': [],
            'chunk_history': [],
        }

        logger.info(f"Initialized RTC policy with overlap={config.overlap_size}, "
                   f"blending={config.blending_method}")

    def reset(self):
        """Reset the RTC state for a new episode."""
        self.current_chunk = None
        self.prev_chunk_tail = None
        self.step_in_chunk = 0
        self.total_steps = 0

        if self.config.track_metrics:
            self.metrics = {
                'boundary_consistency': [],
                'chunk_transitions': [],
                'action_history': [],
                'chunk_history': [],
            }

    def infer(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform RTC inference.

        Args:
            observation: Current observation dict

        Returns:
            Dict with 'actions' key containing the blended action for current step
        """
        # Check if we need to get a new chunk
        need_new_chunk = (
            self.current_chunk is None or
            self.step_in_chunk >= self.config.replan_interval
        )

        if need_new_chunk:
            # Get new chunk from base policy
            result = self.base_policy.infer(observation)
            new_chunk = result['actions']  # Shape: [H, A]

            # Store for metrics
            if self.config.track_metrics:
                self.metrics['chunk_history'].append(new_chunk.copy())

            # Blend with previous chunk if we have overlap
            if self.prev_chunk_tail is not None and self.config.overlap_size > 0:
                blended_chunk = self._blend_chunks(
                    self.prev_chunk_tail,
                    new_chunk[:self.config.overlap_size]
                )
                # Replace the head of new chunk with blended version
                new_chunk[:self.config.overlap_size] = blended_chunk

                # Track boundary consistency
                if self.config.track_metrics:
                    consistency = self._compute_boundary_consistency(
                        self.prev_chunk_tail,
                        new_chunk[:self.config.overlap_size]
                    )
                    self.metrics['boundary_consistency'].append(consistency)
                    self.metrics['chunk_transitions'].append({
                        'step': self.total_steps,
                        'prev_tail': self.prev_chunk_tail.copy(),
                        'new_head': new_chunk[:self.config.overlap_size].copy(),
                        'blended': blended_chunk.copy()
                    })

            # Update state
            self.current_chunk = new_chunk
            self.step_in_chunk = 0

            # Store tail for next blending
            if len(new_chunk) >= self.config.overlap_size:
                self.prev_chunk_tail = new_chunk[-self.config.overlap_size:].copy()

        # Get current action from chunk
        action = self.current_chunk[self.step_in_chunk]

        # Track action history
        if self.config.track_metrics:
            self.metrics['action_history'].append(action.copy())

        # Update counters
        self.step_in_chunk += 1
        self.total_steps += 1

        return {'actions': action}

    def _blend_chunks(self, prev_tail: np.ndarray, new_head: np.ndarray) -> np.ndarray:
        """
        Blend overlapping regions of two chunks.

        Args:
            prev_tail: Tail of previous chunk, shape [O, A]
            new_head: Head of new chunk, shape [O, A]

        Returns:
            Blended actions, shape [O, A]
        """
        O = len(prev_tail)

        if self.config.blending_method == 'none':
            # No blending, just use new chunk
            return new_head

        elif self.config.blending_method == 'linear':
            # Linear interpolation: weight increases linearly from 0 to 1
            weights = np.linspace(0, 1, O)[:, None]  # Shape: [O, 1]
            blended = (1 - weights) * prev_tail + weights * new_head

        elif self.config.blending_method == 'cosine':
            # Cosine interpolation for smoother blending
            t = np.linspace(0, 1, O)
            weights = (1 - np.cos(t * np.pi)) / 2  # Smooth S-curve
            weights = weights[:, None]  # Shape: [O, 1]
            blended = (1 - weights) * prev_tail + weights * new_head

        else:
            raise ValueError(f"Unknown blending method: {self.config.blending_method}")

        return blended

    def _compute_boundary_consistency(
        self,
        prev_tail: np.ndarray,
        new_head: np.ndarray
    ) -> float:
        """
        Compute boundary consistency metric (Bjump from paper).

        Measures the L2 distance between the last action of previous chunk
        and first action of new chunk (before blending).

        Args:
            prev_tail: Tail of previous chunk, shape [O, A]
            new_head: Head of new chunk, shape [O, A]

        Returns:
            Boundary jump magnitude
        """
        # Compare last action of prev with first action of new
        jump = np.linalg.norm(prev_tail[-1] - new_head[0])
        return float(jump)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get collected metrics.

        Returns:
            Dictionary of metrics including:
            - boundary_consistency: List of boundary jumps
            - action_history: Full action sequence
            - chunk_history: All predicted chunks
            - temporal_smoothness: Computed smoothness metrics
        """
        if not self.config.track_metrics:
            return {}

        metrics = self.metrics.copy()

        # Compute temporal smoothness metrics if we have action history
        if len(self.metrics['action_history']) > 0:
            actions = np.array(self.metrics['action_history'])  # Shape: [T, A]

            # First-order smoothness (TV-L1)
            if len(actions) > 1:
                first_order = np.mean(np.abs(actions[1:] - actions[:-1]))
                metrics['tv_l1'] = float(first_order)

            # Second-order smoothness (acceleration)
            if len(actions) > 2:
                second_order = np.mean(
                    np.abs(actions[2:] - 2*actions[1:-1] + actions[:-2])
                )
                metrics['acceleration'] = float(second_order)

            # Global variation (total variation)
            if len(actions) > 1:
                total_var = np.sum(np.abs(actions[1:] - actions[:-1]))
                metrics['total_variation'] = float(total_var)

            # Frequency regularity (via FFT)
            if len(actions) > 10:
                freq_metrics = self._compute_frequency_metrics(actions)
                metrics.update(freq_metrics)

        # Boundary consistency statistics
        if len(self.metrics['boundary_consistency']) > 0:
            bc = np.array(self.metrics['boundary_consistency'])
            metrics['boundary_consistency_stats'] = {
                'mean': float(np.mean(bc)),
                'std': float(np.std(bc)),
                'max': float(np.max(bc)),
                'min': float(np.min(bc)),
            }

        return metrics

    def _compute_frequency_metrics(self, actions: np.ndarray) -> Dict[str, float]:
        """
        Compute frequency-domain metrics.

        Args:
            actions: Action sequence, shape [T, A]

        Returns:
            Dictionary of frequency metrics
        """
        T, A = actions.shape
        metrics = {}

        # Compute FFT for each action dimension
        fft_magnitudes = []
        for a in range(A):
            fft = np.fft.fft(actions[:, a])
            magnitude = np.abs(fft[:T//2])  # Only positive frequencies
            fft_magnitudes.append(magnitude)

        # Average across dimensions
        avg_magnitude = np.mean(fft_magnitudes, axis=0)

        # High-frequency ratio (as in paper)
        # Ratio of high-frequency energy to total energy
        cutoff = len(avg_magnitude) // 4  # Top 25% frequencies
        high_freq_energy = np.sum(avg_magnitude[cutoff:] ** 2)
        total_energy = np.sum(avg_magnitude ** 2)

        if total_energy > 0:
            hf_ratio = high_freq_energy / total_energy
            metrics['high_freq_ratio'] = float(hf_ratio)

        # Dominant frequency
        if len(avg_magnitude) > 0:
            dominant_freq_idx = np.argmax(avg_magnitude)
            metrics['dominant_frequency'] = float(dominant_freq_idx)

        return metrics


class RTCPolicyWrapper:
    """
    Convenience wrapper to create RTC policy from config name and checkpoint.
    """

    @staticmethod
    def create_from_checkpoint(
        config_name: str,
        checkpoint_dir: str,
        rtc_config: Optional[RTCConfig] = None,
        default_prompt: Optional[str] = None,
    ):
        """
        Create RTC policy from checkpoint.

        Args:
            config_name: Training config name (e.g., 'pi05_libero')
            checkpoint_dir: Path to checkpoint directory
            rtc_config: RTC configuration (uses defaults if None)
            default_prompt: Default prompt for policy

        Returns:
            RTCPolicy instance
        """
        from openpi.training import config as _config
        from openpi.policies import policy_config as _policy_config

        # Load base policy
        config = _config.get_config(config_name)
        base_policy = _policy_config.create_trained_policy(
            config,
            checkpoint_dir,
            default_prompt=default_prompt,
        )

        # Create RTC wrapper
        if rtc_config is None:
            rtc_config = RTCConfig()

        return RTCPolicy(base_policy, rtc_config)
