"""Small CALVIN-to-openpi policy adapter with no CALVIN import dependency."""

from collections.abc import Mapping
from typing import Any

import numpy as np

from openpi.policies.rtc_policy import RTCConfig
from openpi.policies.rtc_policy import RTCPolicy

CALVIN_ACTION_DIM = 7


def _as_uint8_hwc(image: Any, *, name: str) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape [H, W, 3] or [3, H, W]")
    if array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] != 3 or array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError(f"{name} must have shape [H, W, 3] or [3, H, W]")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError(f"{name} must contain real numeric pixels")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite pixels")

    if np.issubdtype(array.dtype, np.floating):
        if np.any(array < 0.0) or np.any(array > 1.0):
            raise ValueError(f"floating-point {name} pixels must be within [0, 1]")
        array = (array * 255.0).astype(np.uint8)
    else:
        if np.any(array < 0) or np.any(array > 255):
            raise ValueError(f"integer {name} pixels must be within [0, 255]")
        array = array.astype(np.uint8, copy=False)
    return np.ascontiguousarray(array)


def _as_state(robot_obs: Any) -> np.ndarray:
    state = np.asarray(robot_obs)
    if state.ndim != 1 or state.size == 0:
        raise ValueError("robot_obs must be a non-empty vector")
    if not np.issubdtype(state.dtype, np.number) or np.issubdtype(state.dtype, np.complexfloating):
        raise ValueError("robot_obs must contain real numeric values")
    state = state.astype(np.float32, copy=False)
    if not np.all(np.isfinite(state)):
        raise ValueError("robot_obs must contain only finite values")
    return state.copy()


def _prompt_from_goal(goal: Any) -> str:
    if isinstance(goal, str):
        prompt = goal
    elif isinstance(goal, Mapping):
        prompt = goal.get("lang_text")
    else:
        prompt = None
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("CALVIN goal must be a non-empty string or contain 'lang_text'")
    return prompt


def _calvin_action_chunk(result: Any) -> dict[str, Any]:
    if not isinstance(result, Mapping) or "actions" not in result:
        raise ValueError("policy inference must return a mapping containing 'actions'")
    actions = np.asarray(result["actions"])
    if actions.ndim != 2:
        raise ValueError("base policy actions must have shape [H, D]")
    if actions.shape[0] <= 0:
        raise ValueError("base policy action horizon must be positive")
    if actions.shape[1] != CALVIN_ACTION_DIM:
        raise ValueError(f"CALVIN action chunks must have width {CALVIN_ACTION_DIM}")
    if not np.issubdtype(actions.dtype, np.number) or np.issubdtype(actions.dtype, np.complexfloating):
        raise ValueError("CALVIN actions must contain real numeric values")
    if not np.all(np.isfinite(actions)):
        raise ValueError("CALVIN actions must contain only finite values")

    with np.errstate(over="ignore", invalid="ignore"):
        actions = actions.astype(np.float32, copy=True)
    if not np.all(np.isfinite(actions)):
        raise ValueError("CALVIN actions must remain finite after float32 conversion")
    actions[:, -1] = np.where(actions[:, -1] > 0.0, 1.0, -1.0)
    output = dict(result)
    output["actions"] = actions
    return output


class _CalvinActionChunkPolicy:
    """Validate and discretize complete chunks before RTC observes them."""

    def __init__(self, base_policy: Any) -> None:
        self._base_policy = base_policy

    def infer(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        return _calvin_action_chunk(self._base_policy.infer(observation))


def _binarize_calvin_action(action: np.ndarray) -> np.ndarray:
    output = np.asarray(action).copy()
    output[-1] = 1.0 if output[-1] > 0.0 else -1.0
    return output


class CalvinPolicyAdapter:
    """Expose an openpi chunk policy through CALVIN's ``reset``/``step`` API."""

    def __init__(
        self,
        base_policy: Any,
        *,
        rtc_config: RTCConfig | None = None,
    ) -> None:
        if not hasattr(base_policy, "infer"):
            raise TypeError("base_policy must implement infer(observation)")
        self._base_policy = base_policy
        self._chunk_policy = _CalvinActionChunkPolicy(base_policy)
        self._rtc_policy = (
            RTCPolicy(
                self._chunk_policy,
                rtc_config,
                action_transform=_binarize_calvin_action,
            )
            if rtc_config is not None
            else None
        )
        self._execution_policy = self._rtc_policy or self._chunk_policy

    @property
    def action_dim(self) -> int:
        return CALVIN_ACTION_DIM

    def reset(self) -> None:
        if self._rtc_policy is not None:
            self._rtc_policy.reset()
        reset = getattr(self._base_policy, "reset", None)
        if callable(reset):
            reset()

    def _convert_observation(self, observation: Mapping[str, Any], goal: Any) -> dict[str, Any]:
        try:
            rgb_obs = observation["rgb_obs"]
            static_image = rgb_obs["rgb_static"]
            gripper_image = rgb_obs["rgb_gripper"]
            robot_obs = observation["robot_obs"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "CALVIN observation must contain rgb_obs.rgb_static, "
                "rgb_obs.rgb_gripper, and robot_obs"
            ) from exc

        return {
            "observation/image": _as_uint8_hwc(static_image, name="rgb_static"),
            "observation/wrist_image": _as_uint8_hwc(gripper_image, name="rgb_gripper"),
            "observation/state": _as_state(robot_obs),
            "prompt": _prompt_from_goal(goal),
        }

    def step(self, observation: Mapping[str, Any], goal: Any) -> np.ndarray:
        converted = self._convert_observation(observation, goal)
        result = self._execution_policy.infer(converted)
        actions = np.asarray(result["actions"])
        if self._rtc_policy is None:
            action = actions[0]
        else:
            if actions.ndim != 1:
                raise ValueError("RTC policy action must have shape [D]")
            action = actions

        if action.shape != (CALVIN_ACTION_DIM,):
            raise ValueError(f"CALVIN action must have shape [{CALVIN_ACTION_DIM}]")
        return action.copy()

    def get_metrics(self) -> dict[str, Any]:
        if self._rtc_policy is None:
            return {}
        return self._rtc_policy.get_metrics()
