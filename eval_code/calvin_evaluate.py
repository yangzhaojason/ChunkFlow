"""Synchronous openpi-native evaluation on CALVIN long-horizon sequences.

CALVIN is an optional external benchmark. Importing this module never imports
CALVIN; benchmark modules are loaded only after user-supplied paths are checked.
"""

import argparse
from collections.abc import Callable, Mapping
import dataclasses
import json
import math
import numbers
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np

from eval_code.calvin_policy import CALVIN_ACTION_DIM
from eval_code.calvin_policy import CalvinPolicyAdapter
from openpi.policies.rtc_policy import RTCConfig
from openpi.shared.metrics import summarize_trajectory_metrics


@dataclasses.dataclass(frozen=True)
class CalvinBindings:
    get_env: Callable[..., Any]
    get_sequences: Callable[..., Any]
    get_env_state_for_initial_condition: Callable[[Any], tuple[Any, Any]]
    tasks_class: type


@dataclasses.dataclass(frozen=True)
class ExternalPaths:
    calvin_root: Path
    dataset_root: Path
    checkpoint_dir: Path
    output_dir: Path


@dataclasses.dataclass(frozen=True)
class SequenceResult:
    completed_subtasks: int
    subtask_successes: tuple[bool, ...]
    actions: np.ndarray
    policy_latencies_ms: tuple[float, ...]
    environment_latencies_ms: tuple[float, ...]
    metrics: dict[str, float | int]


def _load_calvin() -> CalvinBindings:
    from calvin_agent.evaluation.multistep_sequences import get_sequences
    from calvin_agent.evaluation.utils import get_env_state_for_initial_condition
    from calvin_env.envs.play_table_env import get_env
    from calvin_env.envs.tasks import Tasks

    return CalvinBindings(
        get_env=get_env,
        get_sequences=get_sequences,
        get_env_state_for_initial_condition=get_env_state_for_initial_condition,
        tasks_class=Tasks,
    )


def require_calvin() -> CalvinBindings:
    """Load optional CALVIN modules with an actionable installation error."""
    try:
        return _load_calvin()
    except ImportError as exc:
        raise RuntimeError(
            "CALVIN is not importable. Install it into the current Python environment, "
            "or set CALVIN_ROOT and use eval_code/eval_calvin_chunkflow.sh."
        ) from exc


def _annotation_for_subtask(annotations: Mapping[str, Any], subtask: str) -> str:
    try:
        value = annotations[subtask]
    except KeyError as exc:
        raise ValueError(f"missing CALVIN language annotation for {subtask!r}") from exc
    if isinstance(value, str):
        annotation = value
    elif isinstance(value, (list, tuple)) and value:
        annotation = value[0]
    else:
        annotation = None
    if not isinstance(annotation, str) or not annotation.strip():
        raise ValueError(f"CALVIN language annotation for {subtask!r} must be non-empty")
    return annotation


def _validated_action(action: Any) -> np.ndarray:
    array = np.asarray(action)
    if array.shape != (CALVIN_ACTION_DIM,):
        raise ValueError(f"CALVIN policy step must return an action with shape [{CALVIN_ACTION_DIM}]")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError("CALVIN policy action must contain real numeric values")
    array = array.astype(np.float32, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError("CALVIN policy action must contain only finite values")
    if array[-1] not in (-1.0, 1.0):
        raise ValueError("CALVIN gripper action must be exactly -1 or 1")
    return array.copy()


def _step_environment(env: Any, action: np.ndarray) -> tuple[Any, Any]:
    result = env.step(action)
    if not isinstance(result, tuple) or len(result) not in (4, 5):
        raise RuntimeError("CALVIN environment step must return a Gym 4- or 5-tuple")
    if len(result) == 4:
        observation, _, _, info = result
    else:
        observation, _, _, _, info = result
    return observation, info


def _seam_pairs(policy: Any) -> list[tuple[np.ndarray, np.ndarray]]:
    get_metrics = getattr(policy, "get_metrics", None)
    if not callable(get_metrics):
        return []
    metrics = get_metrics()
    if not isinstance(metrics, Mapping):
        return []
    pairs = []
    for transition in metrics.get("chunk_transitions", ()):
        if isinstance(transition, Mapping) and "prev_tail" in transition and "new_head" in transition:
            pairs.append((np.asarray(transition["prev_tail"]), np.asarray(transition["new_head"])))
    return pairs


def evaluate_sequence(
    *,
    env: Any,
    policy: Any,
    task_oracle: Any,
    initial_state: Any,
    subtasks: tuple[str, ...] | list[str],
    annotations: Mapping[str, Any],
    state_converter: Callable[[Any], tuple[Any, Any]],
    max_steps: int = 360,
    hf_cutoff_ratio: float = 0.3,
) -> SequenceResult:
    """Evaluate one CALVIN sequence and stop at its first failed subtask."""
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")

    robot_obs, scene_obs = state_converter(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    completed_subtasks = 0
    successes: list[bool] = []
    actions: list[np.ndarray] = []
    policy_latencies_ms: list[float] = []
    environment_latencies_ms: list[float] = []
    subtask_metrics: list[dict[str, Any]] = []

    for subtask in subtasks:
        prompt = _annotation_for_subtask(annotations, subtask)
        reset = getattr(policy, "reset", None)
        if callable(reset):
            reset()
        observation = env.get_obs()
        start_info = env.get_info()
        succeeded = False
        current_actions: list[np.ndarray] = []

        for _ in range(max_steps):
            start = time.perf_counter()
            action = _validated_action(policy.step(observation, prompt))
            latency_ms = (time.perf_counter() - start) * 1000.0
            environment_start = time.perf_counter()
            observation, current_info = _step_environment(env, action)
            environment_latency_ms = (time.perf_counter() - environment_start) * 1000.0
            actions.append(action)
            current_actions.append(action)
            policy_latencies_ms.append(latency_ms)
            environment_latencies_ms.append(environment_latency_ms)

            achieved = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
            if achieved:
                succeeded = True
                break

        if current_actions:
            subtask_metrics.append(
                summarize_trajectory_metrics(
                    np.stack(current_actions),
                    seam_pairs=_seam_pairs(policy),
                    hf_cutoff_ratio=hf_cutoff_ratio,
                )
            )
        successes.append(succeeded)
        if not succeeded:
            break
        completed_subtasks += 1

    if actions:
        action_matrix = np.stack(actions)
    else:
        action_dim = getattr(policy, "action_dim", 7)
        action_matrix = np.empty((0, action_dim), dtype=np.float32)
    if subtask_metrics:
        metric_names = sorted({name for item in subtask_metrics for name in item})
        metrics = {
            name: float(np.mean([float(item[name]) for item in subtask_metrics if name in item]))
            for name in metric_names
        }
    else:
        metrics = summarize_trajectory_metrics(action_matrix, hf_cutoff_ratio=hf_cutoff_ratio)
    metrics["executed_actions"] = len(actions)
    metrics["policy_latency_ms_mean"] = (
        float(math.fsum(policy_latencies_ms) / len(policy_latencies_ms)) if policy_latencies_ms else 0.0
    )
    metrics["environment_latency_ms_mean"] = (
        float(math.fsum(environment_latencies_ms) / len(environment_latencies_ms))
        if environment_latencies_ms
        else 0.0
    )
    return SequenceResult(
        completed_subtasks=completed_subtasks,
        subtask_successes=tuple(successes),
        actions=action_matrix,
        policy_latencies_ms=tuple(policy_latencies_ms),
        environment_latencies_ms=tuple(environment_latencies_ms),
        metrics=metrics,
    )


def compute_success_rates(results: list[int] | tuple[int, ...]) -> tuple[float, float, float, float, float]:
    """Return CALVIN success rates for completing at least 1 through 5 tasks."""
    if not results:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    values = []
    for value in results:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 0 <= int(value) <= 5:
            raise ValueError("CALVIN sequence results must be integers within [0, 5]")
        values.append(int(value))
    count = len(values)
    return tuple(sum(value >= length for value in values) / count for length in range(1, 6))


def select_evaluation_sequences(
    get_sequences: Callable[[int], Any],
    *,
    total_sequences: int,
    start_index: int,
) -> list[Any]:
    """Generate one canonical CALVIN set, then take its resumable suffix."""
    sequences = list(get_sequences(total_sequences))
    if len(sequences) != total_sequences:
        raise RuntimeError(
            f"CALVIN returned {len(sequences)} sequences, expected {total_sequences}"
        )
    return sequences[start_index:]


def evaluate_policy(
    *,
    env: Any,
    policy: Any,
    task_oracle: Any,
    sequences: list[tuple[Any, Any]],
    annotations: Mapping[str, Any],
    state_converter: Callable[[Any], tuple[Any, Any]],
    max_steps: int,
    hf_cutoff_ratio: float,
    sequence_start_index: int = 0,
) -> tuple[dict[str, Any], list[SequenceResult]]:
    sequence_results = []
    for index, (initial_state, subtasks) in enumerate(sequences):
        result = evaluate_sequence(
            env=env,
            policy=policy,
            task_oracle=task_oracle,
            initial_state=initial_state,
            subtasks=tuple(subtasks),
            annotations=annotations,
            state_converter=state_converter,
            max_steps=max_steps,
            hf_cutoff_ratio=hf_cutoff_ratio,
        )
        sequence_results.append(result)
        rates = compute_success_rates(tuple(item.completed_subtasks for item in sequence_results))
        description = " | ".join(f"{length}/5: {rate * 100:.1f}%" for length, rate in enumerate(rates, 1))
        actual_index = sequence_start_index + index
        print(f"[sequence {actual_index}] {description}")

    completed = [result.completed_subtasks for result in sequence_results]
    rates = compute_success_rates(tuple(completed))
    metric_names = sorted(
        {
            name
            for result in sequence_results
            for name in result.metrics
            if name != "executed_actions"
        }
    )
    aggregate_metrics = {}
    for name in metric_names:
        values = [float(result.metrics[name]) for result in sequence_results if name in result.metrics]
        aggregate_metrics[name] = float(np.mean(values)) if values else 0.0
    summary = {
        "num_sequences": len(sequence_results),
        "total_executed_actions": sum(len(result.actions) for result in sequence_results),
        "mean_executed_actions_per_sequence": (
            float(np.mean([len(result.actions) for result in sequence_results]))
            if sequence_results
            else 0.0
        ),
        "average_completed_subtasks": float(np.mean(completed)) if completed else 0.0,
        "success_rates": {f"{length}/5": rate for length, rate in enumerate(rates, 1)},
        "metrics": aggregate_metrics,
        "sequences": [
            {
                "index": sequence_start_index + index,
                "subtasks": list(sequences[index][1]),
                "completed_subtasks": result.completed_subtasks,
                "subtask_successes": list(result.subtask_successes),
                "executed_actions": len(result.actions),
                "metrics": result.metrics,
            }
            for index, result in enumerate(sequence_results)
        ],
    }
    return summary, sequence_results


def _required_directory(value: str, *, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be provided")
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{name} is not a directory: {path}")
    return path


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def validate_external_paths(args: argparse.Namespace) -> ExternalPaths:
    """Validate all external resources before loading a policy or environment."""
    calvin_root = _required_directory(args.calvin_root, name="calvin_root")
    dataset_root = _required_directory(args.dataset_path, name="dataset_path")
    checkpoint_dir = _required_directory(args.checkpoint_dir, name="checkpoint_dir")
    expected = (
        calvin_root / "calvin_models" / "calvin_agent",
        calvin_root / "calvin_env" / "calvin_env",
        dataset_root / "validation",
        checkpoint_dir / "assets",
    )
    missing = [str(path) for path in expected if not path.is_dir()]
    if missing:
        raise FileNotFoundError("required CALVIN directories are missing: " + ", ".join(missing))
    required_files = (
        dataset_root / "validation" / ".hydra" / "merged_config.yaml",
        calvin_root
        / "calvin_models"
        / "conf"
        / "callbacks"
        / "rollout"
        / "tasks"
        / "new_playtable_tasks.yaml",
        calvin_root / "calvin_models" / "conf" / "annotations" / "new_playtable_validation.yaml",
    )
    missing_files = [str(path) for path in required_files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError("required CALVIN files are missing: " + ", ".join(missing_files))
    if not (checkpoint_dir / "params").is_dir() and not (checkpoint_dir / "model.safetensors").is_file():
        raise FileNotFoundError("checkpoint_dir must contain params/ or model.safetensors")
    if not isinstance(args.config_name, str) or not args.config_name.strip():
        raise ValueError("config_name must be provided")
    _positive_integer(args.num_sequences, name="num_sequences")
    _positive_integer(args.max_steps, name="max_steps")
    if args.action_dim != CALVIN_ACTION_DIM:
        raise ValueError(f"action_dim must be exactly {CALVIN_ACTION_DIM} for cartesian CALVIN evaluation")
    if isinstance(args.start_index, bool) or not isinstance(args.start_index, numbers.Integral) or args.start_index < 0:
        raise ValueError("start_index must be a non-negative integer")
    if args.start_index > args.num_sequences:
        raise ValueError("start_index must not exceed num_sequences")
    if not args.disable_rtc:
        _positive_integer(args.replan_interval, name="replan_interval")
        if (
            isinstance(args.overlap_size, bool)
            or not isinstance(args.overlap_size, numbers.Integral)
            or args.overlap_size < 0
        ):
            raise ValueError("overlap_size must be a non-negative integer")
        if args.blending_method not in {"none", "linear", "cosine"}:
            raise ValueError("blending_method must be one of: none, linear, cosine")
    if (
        isinstance(args.hf_cutoff_ratio, bool)
        or not isinstance(args.hf_cutoff_ratio, numbers.Real)
        or not math.isfinite(float(args.hf_cutoff_ratio))
        or not 0.0 <= float(args.hf_cutoff_ratio) <= 1.0
    ):
        raise ValueError("hf_cutoff_ratio must be within [0, 1]")
    if not isinstance(args.output_dir, str) or not args.output_dir.strip():
        raise ValueError("output_dir must be provided")
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parents[1] / output_dir
    return ExternalPaths(
        calvin_root=calvin_root,
        dataset_root=dataset_root,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir.resolve(),
    )


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read the official CALVIN configuration") from exc
    with path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected a mapping in CALVIN configuration: {path}")
    return payload


def _load_calvin_protocol(paths: ExternalPaths, bindings: CalvinBindings) -> tuple[Any, Mapping[str, Any]]:
    config_root = paths.calvin_root / "calvin_models" / "conf"
    task_config = _load_yaml(config_root / "callbacks" / "rollout" / "tasks" / "new_playtable_tasks.yaml")
    annotations = _load_yaml(config_root / "annotations" / "new_playtable_validation.yaml")
    tasks = task_config.get("tasks")
    if not isinstance(tasks, Mapping):
        raise ValueError("official CALVIN task configuration has no 'tasks' mapping")
    return bindings.tasks_class(tasks=tasks), annotations


def _create_policy(args: argparse.Namespace, checkpoint_dir: Path) -> CalvinPolicyAdapter:
    from openpi.policies import policy_config
    from openpi.training import config as training_config

    config = training_config.get_config(args.config_name)
    base_policy = policy_config.create_trained_policy(
        config,
        checkpoint_dir,
        pytorch_device=args.pytorch_device,
    )
    rtc_config = None
    if not args.disable_rtc:
        rtc_config = RTCConfig.from_model_config(
            config.model,
            overlap_size=args.overlap_size,
            blending_method=args.blending_method,
            replan_interval=args.replan_interval,
            track_metrics=True,
        )
    return CalvinPolicyAdapter(base_policy, rtc_config=rtc_config)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _prepare_output_directory(output_dir: Path, *, save_actions: bool) -> None:
    """Create and write-check every output directory before rollout."""
    directories = [output_dir]
    if save_actions:
        directories.append(output_dir / "actions")
    for directory in directories:
        if directory.exists() and not directory.is_dir():
            raise NotADirectoryError(f"output path is not a directory: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.NamedTemporaryFile(prefix=".chunkflow-write-test-", dir=directory):
                pass
        except OSError as exc:
            raise PermissionError(f"output directory is not writable: {directory}") from exc


def _write_results(
    output_dir: Path,
    summary: Mapping[str, Any],
    sequence_results: list[SequenceResult],
    *,
    save_actions: bool,
    sequence_start_index: int = 0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_actions:
        actions_dir = output_dir / "actions"
        actions_dir.mkdir(parents=True, exist_ok=True)
        for offset, result in enumerate(sequence_results):
            index = sequence_start_index + offset
            np.savez_compressed(actions_dir / f"sequence_{index:06d}.npz", pred_actions=result.actions)
    result_end_index = sequence_start_index + len(sequence_results)
    results_path = output_dir / (
        f"results_{sequence_start_index:06d}_{result_end_index:06d}.json"
    )
    with results_path.open("w", encoding="utf-8") as stream:
        json.dump(_json_safe(summary), stream, indent=2, sort_keys=True)
        stream.write("\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an openpi ChunkFlow policy on CALVIN.")
    parser.add_argument("--calvin-root", default="", help="Path to the official CALVIN checkout.")
    parser.add_argument("--dataset-path", default="", help="Path to a CALVIN dataset root containing validation/.")
    parser.add_argument("--checkpoint-dir", default="", help="Path to an openpi checkpoint directory.")
    parser.add_argument("--config-name", default="", help="openpi training config used by the checkpoint.")
    parser.add_argument("--output-dir", default="outputs/calvin")
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=1000,
        help="Size of the canonical CALVIN sequence set before --start-index slicing.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=360)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--hf-cutoff-ratio", type=float, default=0.3)
    parser.add_argument("--pytorch-device", default=None)
    parser.add_argument("--show-gui", action="store_true")
    parser.add_argument("--save-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-rtc", action="store_true")
    parser.add_argument("--replan-interval", type=int, default=5)
    parser.add_argument("--overlap-size", type=int, default=4)
    parser.add_argument("--blending-method", choices=("none", "linear", "cosine"), default="linear")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = validate_external_paths(args)
    _prepare_output_directory(paths.output_dir, save_actions=args.save_actions)
    bindings = require_calvin()
    sequences = select_evaluation_sequences(
        bindings.get_sequences,
        total_sequences=args.num_sequences,
        start_index=args.start_index,
    )
    task_oracle, annotations = _load_calvin_protocol(paths, bindings)
    policy = _create_policy(args, paths.checkpoint_dir)
    env = bindings.get_env(paths.dataset_root / "validation", show_gui=args.show_gui)
    summary, sequence_results = evaluate_policy(
        env=env,
        policy=policy,
        task_oracle=task_oracle,
        sequences=sequences,
        annotations=annotations,
        state_converter=bindings.get_env_state_for_initial_condition,
        max_steps=args.max_steps,
        hf_cutoff_ratio=args.hf_cutoff_ratio,
        sequence_start_index=args.start_index,
    )
    summary["configuration"] = {
        "config_name": args.config_name,
        "rtc_enabled": not args.disable_rtc,
        "replan_interval": args.replan_interval,
        "overlap_size": args.overlap_size,
        "blending_method": args.blending_method,
        "max_steps": args.max_steps,
    }
    _write_results(
        paths.output_dir,
        summary,
        sequence_results,
        save_actions=args.save_actions,
        sequence_start_index=args.start_index,
    )
    print(f"Wrote CALVIN results to {paths.output_dir}")
    return summary


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
