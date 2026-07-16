from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, List, Tuple, Dict, Any
from collections import deque

import numpy as np

from openpi.policies.policy import Policy


@dataclass
class RTCConfig:
    # Number of steps predicted per chunk by the policy (action horizon)
    action_horizon: int
    # Replan cadence: how many steps to execute before requesting a new chunk
    replan_steps: int
    # Blend across overlap between previous tail and new head
    enable_overlap_blend: bool = True
    # Blend schedule: 'linear' currently
    blend_schedule: str = "linear"

    def __post_init__(self) -> None:
        if self.replan_steps <= 0 or self.action_horizon <= 0:
            raise ValueError("action_horizon and replan_steps must be positive")
        if self.replan_steps > self.action_horizon:
            raise ValueError("replan_steps must be <= action_horizon")


class RealTimeChunkExecutor:
    """
    Real-Time Execution broker for action chunking flow policies (RTC).
    - Queries the wrapped policy intermittently (every `replan_steps`) for a new action chunk.
    - Optionally blends overlapping steps between the previous chunk (tail) and the new chunk (head)
      to improve boundary (seam) consistency.
    - Returns one action per call to `step`.
    """

    def __init__(self, policy: Policy, config: RTCConfig):
        self._policy = policy
        self._cfg = config
        self._buffer: Deque[np.ndarray] = deque()
        self._prev_full_chunk: np.ndarray | None = None
        self._steps_since_plan: int = 0
        self._seam_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        self._executed_actions: List[np.ndarray] = []

    @property
    def overlap(self) -> int:
        return self._cfg.action_horizon - self._cfg.replan_steps

    def reset(self) -> None:
        self._buffer.clear()
        self._prev_full_chunk = None
        self._steps_since_plan = 0
        self._seam_pairs.clear()
        self._executed_actions.clear()

    def _blend_overlap(self, prev_tail: np.ndarray, new_head: np.ndarray) -> np.ndarray:
        """
        Blend overlap region of length O between previous tail and new head.
        Linear schedule: a_t = w * prev + (1-w) * new, where w descends from 1->0.
        """
        O = prev_tail.shape[0]
        if O == 0:
            return new_head
        if self._cfg.blend_schedule == "linear":
            # Weights from 1 -> 0 across overlap
            ws = np.linspace(1.0, 0.0, O, dtype=np.float64)[:, None]
            return ws * prev_tail.astype(np.float64) + (1.0 - ws) * new_head.astype(np.float64)
        # Default to linear if unknown
        ws = np.linspace(1.0, 0.0, O, dtype=np.float64)[:, None]
        return ws * prev_tail.astype(np.float64) + (1.0 - ws) * new_head.astype(np.float64)

    def _request_new_chunk(self, obs: Dict[str, Any]) -> None:
        """
        Query policy for a new chunk and populate the buffer with blended actions if overlap applies.
        """
        result = self._policy.infer(obs)
        chunk: np.ndarray = result["actions"]  # [H, D]
        if not isinstance(chunk, np.ndarray):
            chunk = np.asarray(chunk)
        assert chunk.ndim == 2, "policy must output actions with shape [H, D]"
        H = chunk.shape[0]
        if H < self._cfg.replan_steps:
            raise ValueError(
                f"Policy predicted {H} steps, but replan_steps={self._cfg.replan_steps}"
            )
        O = self.overlap
        self._buffer.clear()

        if self._cfg.enable_overlap_blend and self._prev_full_chunk is not None and O > 0:
            prev_tail = self._prev_full_chunk[self._cfg.replan_steps : self._cfg.action_horizon]
            new_head = chunk[:O]
            # Save seam pair for metrics
            self._seam_pairs.append((prev_tail.copy(), new_head.copy()))
            blended = self._blend_overlap(prev_tail, new_head)
            # Fill buffer: blended overlap region, then the remaining fresh steps up to replan_steps
            for a in blended:
                self._buffer.append(np.asarray(a, dtype=chunk.dtype))
            remainder = chunk[O:self._cfg.replan_steps]
            for a in remainder:
                self._buffer.append(np.asarray(a, dtype=chunk.dtype))
        else:
            # No blending: simply push first `replan_steps` steps
            for a in chunk[: self._cfg.replan_steps]:
                self._buffer.append(np.asarray(a, dtype=chunk.dtype))

        # Keep original full chunk to enable overlap with next plan
        self._prev_full_chunk = chunk
        self._steps_since_plan = 0

    def step(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        Get the next action using RTC. Will trigger re-planning when buffer is empty.
        """
        if not self._buffer:
            self._request_new_chunk(obs)

        action = self._buffer.popleft()
        self._steps_since_plan += 1
        self._executed_actions.append(action.copy())
        return action

    def get_episode_records(self) -> Dict[str, Any]:
        """
        Return and keep accumulated episode records.
        - executed_actions: [T, D]
        - seam_pairs: list of (prev_tail[O,D], new_head[O,D])
        """
        executed = np.asarray(self._executed_actions) if self._executed_actions else np.zeros((0, 0))
        return {
            "executed_actions": executed,
            "seam_pairs": [(p.copy(), n.copy()) for p, n in self._seam_pairs],
        }
