import builtins
import importlib
import os
from pathlib import Path
import shlex
import subprocess
import sys
import types

import numpy as np
import pytest


@pytest.mark.parametrize(
    "module_name",
    ["eval_code.calvin_policy", "eval_code.calvin_evaluate"],
)
def test_calvin_modules_import_without_installed_benchmark(module_name, monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "calvin_agent" or name.startswith("calvin_agent."):
            raise AssertionError("CALVIN was imported eagerly")
        if name == "calvin_env" or name.startswith("calvin_env."):
            raise AssertionError("CALVIN was imported eagerly")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    sys.modules.pop(module_name, None)
    importlib.import_module(module_name)


def test_missing_calvin_dependency_has_actionable_error(monkeypatch):
    module = importlib.import_module("eval_code.calvin_evaluate")
    monkeypatch.setattr(
        module,
        "_load_calvin",
        lambda: (_ for _ in ()).throw(ImportError("calvin_agent")),
    )
    with pytest.raises(RuntimeError, match="CALVIN"):
        module.require_calvin()


class _FakeChunkPolicy:
    def __init__(self, chunks):
        self.chunks = [np.asarray(chunk) for chunk in chunks]
        self.observations = []
        self.reset_calls = 0

    def infer(self, observation):
        self.observations.append(observation)
        index = min(len(self.observations) - 1, len(self.chunks) - 1)
        return {"actions": self.chunks[index].copy()}

    def reset(self):
        self.reset_calls += 1


def _calvin_observation():
    return {
        "rgb_obs": {
            "rgb_static": np.full((3, 2, 4), 0.5, dtype=np.float32),
            "rgb_gripper": np.full((2, 4, 3), 17, dtype=np.uint8),
        },
        "robot_obs": np.arange(15, dtype=np.float64),
    }


def test_policy_adapter_converts_calvin_observation_and_returns_exact_action_shape():
    module = importlib.import_module("eval_code.calvin_policy")
    chunk = np.arange(21, dtype=np.float32).reshape(3, 7)
    base_policy = _FakeChunkPolicy([chunk])
    adapter = module.CalvinPolicyAdapter(base_policy)

    adapter.reset()
    action = adapter.step(_calvin_observation(), "pull the handle")

    assert action.shape == (7,)
    np.testing.assert_array_equal(action, [0, 1, 2, 3, 4, 5, 1])
    assert chunk[0, -1] == 6
    assert base_policy.reset_calls == 1
    converted = base_policy.observations[0]
    assert set(converted) == {
        "observation/image",
        "observation/wrist_image",
        "observation/state",
        "prompt",
    }
    assert converted["observation/image"].shape == (2, 4, 3)
    assert converted["observation/image"].dtype == np.uint8
    np.testing.assert_array_equal(converted["observation/image"], 127)
    np.testing.assert_array_equal(
        converted["observation/wrist_image"],
        _calvin_observation()["rgb_obs"]["rgb_gripper"],
    )
    assert converted["observation/state"].dtype == np.float32
    assert converted["prompt"] == "pull the handle"


def test_policy_adapter_accepts_calvin_goal_mapping():
    module = importlib.import_module("eval_code.calvin_policy")
    base_policy = _FakeChunkPolicy([np.zeros((2, 7), dtype=np.float32)])
    adapter = module.CalvinPolicyAdapter(base_policy)

    adapter.step(_calvin_observation(), {"lang_text": "switch on the light"})

    assert base_policy.observations[0]["prompt"] == "switch on the light"


@pytest.mark.parametrize(("gripper", "expected"), [(-0.01, -1.0), (0.0, -1.0), (0.01, 1.0)])
def test_policy_adapter_binarizes_gripper_before_execution(gripper, expected):
    module = importlib.import_module("eval_code.calvin_policy")
    chunk = np.zeros((2, 7), dtype=np.float32)
    chunk[:, -1] = gripper
    adapter = module.CalvinPolicyAdapter(_FakeChunkPolicy([chunk]))

    action = adapter.step(_calvin_observation(), "test")

    assert action[-1] == expected
    assert np.all(chunk[:, -1] == gripper)


@pytest.mark.parametrize(
    "chunk, error",
    [
        (np.zeros(7, dtype=np.float32), r"\[H, D\]"),
        (np.zeros((2, 8), dtype=np.float32), "7"),
        (np.full((2, 7), np.nan, dtype=np.float32), "finite"),
        (
            np.vstack(
                [
                    np.zeros((1, 7), dtype=np.float32),
                    np.full((1, 7), np.nan, dtype=np.float32),
                ]
            ),
            "finite",
        ),
        (np.full((2, 7), np.finfo(np.float64).max), "finite"),
    ],
)
def test_policy_adapter_rejects_invalid_policy_actions(chunk, error):
    module = importlib.import_module("eval_code.calvin_policy")
    adapter = module.CalvinPolicyAdapter(_FakeChunkPolicy([chunk]))

    with pytest.raises(ValueError, match=error):
        adapter.step(_calvin_observation(), "test")


def test_policy_adapter_uses_canonical_rtc_policy():
    module = importlib.import_module("eval_code.calvin_policy")
    rtc_module = importlib.import_module("openpi.policies.rtc_policy")
    first = np.arange(28, dtype=np.float32).reshape(4, 7)
    second = first + 100
    base_policy = _FakeChunkPolicy([first, second])
    adapter = module.CalvinPolicyAdapter(
        base_policy,
        rtc_config=rtc_module.RTCConfig(
            replan_interval=2,
            overlap_size=1,
            blending_method="linear",
        ),
    )

    actions = [adapter.step(_calvin_observation(), "test") for _ in range(3)]

    assert len(base_policy.observations) == 2
    assert all(action.shape == (7,) for action in actions)
    np.testing.assert_array_equal(actions[0][:-1], first[0][:-1])
    np.testing.assert_array_equal(actions[1][:-1], first[1][:-1])
    np.testing.assert_array_equal(actions[2][:-1], first[2][:-1])
    assert {float(action[-1]) for action in actions} <= {-1.0, 1.0}
    metrics = adapter.get_metrics()
    assert {float(action[-1]) for action in metrics["action_history"]} <= {-1.0, 1.0}


def test_rtc_binarizes_gripper_after_overlap_blending():
    module = importlib.import_module("eval_code.calvin_policy")
    rtc_module = importlib.import_module("openpi.policies.rtc_policy")
    first = np.zeros((5, 7), dtype=np.float32)
    first[:, -1] = 1.0
    second = np.zeros((5, 7), dtype=np.float32)
    second[:, -1] = -1.0
    adapter = module.CalvinPolicyAdapter(
        _FakeChunkPolicy([first, second]),
        rtc_config=rtc_module.RTCConfig(
            replan_interval=2,
            overlap_size=3,
            blending_method="linear",
        ),
    )

    actions = [adapter.step(_calvin_observation(), "test") for _ in range(4)]

    assert [float(action[-1]) for action in actions] == [1.0, 1.0, 1.0, -1.0]
    assert {float(action[-1]) for action in adapter.get_metrics()["action_history"]} <= {-1.0, 1.0}


def test_calvin_policy_derives_rtc_history_length_from_training_config(monkeypatch, tmp_path):
    module = importlib.import_module("eval_code.calvin_evaluate")
    policy_config = importlib.import_module("openpi.policies.policy_config")
    training_config = importlib.import_module("openpi.training.config")
    model_config = type("ModelConfig", (), {"history_length": 3})()
    train_config = type("TrainConfig", (), {"model": model_config})()
    base_policy = _FakeChunkPolicy([np.zeros((4, 7), dtype=np.float32)])
    monkeypatch.setattr(training_config, "get_config", lambda name: train_config)
    monkeypatch.setattr(policy_config, "create_trained_policy", lambda *args, **kwargs: base_policy)
    args = types.SimpleNamespace(
        config_name="chunkflow",
        pytorch_device=None,
        disable_rtc=False,
        overlap_size=2,
        blending_method="linear",
        replan_interval=2,
    )

    adapter = module._create_policy(args, tmp_path)  # noqa: SLF001

    assert adapter._rtc_policy.config.history_length == 3  # noqa: SLF001


class _FakeEnvironment:
    def __init__(self):
        self.reset_calls = []
        self.step_count = 0
        self.actions = []

    def reset(self, *, robot_obs, scene_obs):
        self.reset_calls.append((robot_obs, scene_obs))

    def get_obs(self):
        return _calvin_observation()

    def get_info(self):
        return {"step": self.step_count}

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        self.step_count += 1
        return self.get_obs(), 0.0, False, {"step": self.step_count}


class _FakeStepPolicy:
    def __init__(self):
        self.reset_calls = 0
        self.prompts = []

    def reset(self):
        self.reset_calls += 1

    def step(self, observation, prompt):
        del observation
        self.prompts.append(prompt)
        action = np.zeros(7, dtype=np.float32)
        action[-1] = 1.0
        return action


class _FakeTaskOracle:
    def get_task_info_for_set(self, start_info, current_info, task_filter):
        del start_info
        subtask = next(iter(task_filter))
        if subtask == "first" and current_info["step"] >= 1:
            return {subtask}
        return set()


def test_evaluate_sequence_resets_per_subtask_and_stops_after_first_failure():
    module = importlib.import_module("eval_code.calvin_evaluate")
    env = _FakeEnvironment()
    policy = _FakeStepPolicy()
    initial_state = {"state": "initial"}
    annotations = {
        "first": ["do the first task"],
        "second": ["do the second task"],
        "third": ["must not run"],
    }

    result = module.evaluate_sequence(
        env=env,
        policy=policy,
        task_oracle=_FakeTaskOracle(),
        initial_state=initial_state,
        subtasks=("first", "second", "third"),
        annotations=annotations,
        state_converter=lambda state: (f"robot:{state['state']}", "scene"),
        max_steps=2,
    )

    assert result.completed_subtasks == 1
    assert result.actions.shape == (3, 7)
    assert result.subtask_successes == (True, False)
    assert env.reset_calls == [("robot:initial", "scene")]
    assert policy.reset_calls == 2
    assert policy.prompts == [
        "do the first task",
        "do the second task",
        "do the second task",
    ]
    assert all(action.shape == (7,) for action in env.actions)
    assert result.metrics["executed_actions"] == 3
    assert len(result.environment_latencies_ms) == 3
    assert result.metrics["environment_latency_ms_mean"] >= 0.0


def test_evaluate_sequence_rejects_non_calvin_action_width():
    module = importlib.import_module("eval_code.calvin_evaluate")

    class WrongWidthPolicy(_FakeStepPolicy):
        def step(self, observation, prompt):
            del observation, prompt
            return np.zeros(8, dtype=np.float32)

    with pytest.raises(ValueError, match=r"\[7\]"):
        module.evaluate_sequence(
            env=_FakeEnvironment(),
            policy=WrongWidthPolicy(),
            task_oracle=_FakeTaskOracle(),
            initial_state={},
            subtasks=("first",),
            annotations={"first": ["first"]},
            state_converter=lambda state: (state, state),
            max_steps=1,
        )


def test_evaluate_sequence_rejects_non_binary_gripper_action():
    module = importlib.import_module("eval_code.calvin_evaluate")

    class ContinuousGripperPolicy(_FakeStepPolicy):
        def step(self, observation, prompt):
            del observation, prompt
            return np.zeros(7, dtype=np.float32)

    with pytest.raises(ValueError, match="gripper"):
        module.evaluate_sequence(
            env=_FakeEnvironment(),
            policy=ContinuousGripperPolicy(),
            task_oracle=_FakeTaskOracle(),
            initial_state={},
            subtasks=("first",),
            annotations={"first": ["first"]},
            state_converter=lambda state: (state, state),
            max_steps=1,
        )


def test_trajectory_metrics_do_not_cross_policy_reset_boundaries():
    module = importlib.import_module("eval_code.calvin_evaluate")

    class ChangingPolicy(_FakeStepPolicy):
        def step(self, observation, prompt):
            del observation, prompt
            action = np.full(7, 100.0 * (self.reset_calls - 1), dtype=np.float32)
            action[-1] = 1.0
            return action

    class AlwaysSuccessfulOracle:
        def get_task_info_for_set(self, start_info, current_info, task_filter):
            del start_info, current_info
            return task_filter

    result = module.evaluate_sequence(
        env=_FakeEnvironment(),
        policy=ChangingPolicy(),
        task_oracle=AlwaysSuccessfulOracle(),
        initial_state={},
        subtasks=("first", "second"),
        annotations={"first": ["first"], "second": ["second"]},
        state_converter=lambda state: (state, state),
        max_steps=1,
    )

    assert result.completed_subtasks == 2
    assert result.metrics["msd_delta_1"] == 0.0
    assert result.metrics["tv_l1"] == 0.0


def test_success_rates_follow_calvin_long_horizon_definition():
    module = importlib.import_module("eval_code.calvin_evaluate")

    assert module.compute_success_rates([0, 1, 3, 5]) == (0.75, 0.5, 0.5, 0.25, 0.25)


def test_sequence_selection_generates_fixed_total_before_slicing():
    module = importlib.import_module("eval_code.calvin_evaluate")
    calls = []

    def get_sequences(total):
        calls.append(total)
        return list(range(total))

    selected = module.select_evaluation_sequences(
        get_sequences,
        total_sequences=1000,
        start_index=500,
    )

    assert calls == [1000]
    assert selected == list(range(500, 1000))


def test_argument_defaults_are_portable_and_validation_precedes_calvin_import(monkeypatch):
    module = importlib.import_module("eval_code.calvin_evaluate")
    args = module.build_arg_parser().parse_args([])
    called = False

    def unexpected_import():
        nonlocal called
        called = True
        raise AssertionError("CALVIN import ran before path validation")

    monkeypatch.setattr(module, "require_calvin", unexpected_import)
    with pytest.raises(ValueError, match="calvin_root"):
        module.run(args)

    assert called is False
    assert args.calvin_root == ""
    assert args.dataset_path == ""
    assert args.checkpoint_dir == ""
    assert args.output_dir == "outputs/calvin"


def _valid_external_args(module, tmp_path):
    calvin_root = tmp_path / "calvin"
    (calvin_root / "calvin_models" / "calvin_agent").mkdir(parents=True)
    (calvin_root / "calvin_env" / "calvin_env").mkdir(parents=True)
    task_config = calvin_root / "calvin_models" / "conf" / "callbacks" / "rollout" / "tasks"
    task_config.mkdir(parents=True)
    (task_config / "new_playtable_tasks.yaml").write_text("tasks: {}\n", encoding="utf-8")
    annotations = calvin_root / "calvin_models" / "conf" / "annotations"
    annotations.mkdir(parents=True)
    (annotations / "new_playtable_validation.yaml").write_text("{}\n", encoding="utf-8")
    dataset_root = tmp_path / "dataset"
    dataset_config = dataset_root / "validation" / ".hydra"
    dataset_config.mkdir(parents=True)
    (dataset_config / "merged_config.yaml").write_text("env: {}\n", encoding="utf-8")
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "params").mkdir()
    (checkpoint_dir / "assets").mkdir()
    return module.build_arg_parser().parse_args(
        [
            "--calvin-root",
            str(calvin_root),
            "--dataset-path",
            str(dataset_root),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--config-name",
            "test_config",
        ]
    )


def test_default_output_directory_is_relative_to_repository(tmp_path, monkeypatch):
    module = importlib.import_module("eval_code.calvin_evaluate")
    args = _valid_external_args(module, tmp_path)
    caller_directory = tmp_path / "caller"
    caller_directory.mkdir()
    monkeypatch.chdir(caller_directory)

    paths = module.validate_external_paths(args)

    repository_root = Path(module.__file__).resolve().parents[1]
    assert paths.output_dir == repository_root / "outputs" / "calvin"


def test_missing_calvin_dataset_configuration_fails_before_optional_import(tmp_path, monkeypatch):
    module = importlib.import_module("eval_code.calvin_evaluate")
    args = _valid_external_args(module, tmp_path)
    (Path(args.dataset_path) / "validation" / ".hydra" / "merged_config.yaml").unlink()
    called = False

    def unexpected_import():
        nonlocal called
        called = True
        raise AssertionError("optional import ran before dataset validation")

    monkeypatch.setattr(module, "require_calvin", unexpected_import)
    with pytest.raises(FileNotFoundError, match="merged_config"):
        module.run(args)
    assert called is False


def test_output_directory_is_prepared_before_optional_import(tmp_path, monkeypatch):
    module = importlib.import_module("eval_code.calvin_evaluate")
    args = _valid_external_args(module, tmp_path)
    output_dir = tmp_path / "new-output"
    args.output_dir = str(output_dir)

    def stop_after_preflight():
        assert output_dir.is_dir()
        raise RuntimeError("stopped after output preflight")

    monkeypatch.setattr(module, "require_calvin", stop_after_preflight)
    with pytest.raises(RuntimeError, match="stopped after output preflight"):
        module.run(args)


def test_action_output_directory_is_prepared_before_optional_import(tmp_path, monkeypatch):
    module = importlib.import_module("eval_code.calvin_evaluate")
    args = _valid_external_args(module, tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "actions").write_text("not a directory\n", encoding="utf-8")
    args.output_dir = str(output_dir)

    monkeypatch.setattr(
        module,
        "require_calvin",
        lambda: (_ for _ in ()).throw(AssertionError("optional import ran after bad action output")),
    )
    with pytest.raises(NotADirectoryError, match="actions"):
        module.run(args)


def test_sequences_are_generated_before_policy_and_environment_initialization(tmp_path, monkeypatch):
    module = importlib.import_module("eval_code.calvin_evaluate")
    args = _valid_external_args(module, tmp_path)
    args.num_sequences = 2
    args.output_dir = str(tmp_path / "output")
    events = []

    def get_sequences(total):
        events.append("sequences")
        return [({}, ("first",)) for _ in range(total)]

    def get_env(*args, **kwargs):
        del args, kwargs
        events.append("environment")
        return object()

    bindings = types.SimpleNamespace(
        get_sequences=get_sequences,
        get_env=get_env,
        get_env_state_for_initial_condition=lambda state: (state, state),
    )
    monkeypatch.setattr(module, "require_calvin", lambda: bindings)
    monkeypatch.setattr(module, "_load_calvin_protocol", lambda paths, loaded: (object(), {}))

    def create_policy(*args, **kwargs):
        del args, kwargs
        events.append("policy")
        return object()

    monkeypatch.setattr(module, "_create_policy", create_policy)
    monkeypatch.setattr(module, "evaluate_policy", lambda **kwargs: ({}, []))
    monkeypatch.setattr(module, "_write_results", lambda *args, **kwargs: None)

    module.run(args)

    assert events == ["sequences", "policy", "environment"]


@pytest.mark.parametrize(
    ("attribute", "value", "error"),
    [
        ("max_steps", 0, "max_steps"),
        ("action_dim", 8, "action_dim.*7"),
        ("start_index", 1001, "start_index.*num_sequences"),
        ("replan_interval", 0, "replan_interval"),
        ("overlap_size", -1, "overlap_size"),
    ],
)
def test_invalid_runtime_settings_fail_before_optional_import(
    attribute,
    value,
    error,
    tmp_path,
    monkeypatch,
):
    module = importlib.import_module("eval_code.calvin_evaluate")
    args = _valid_external_args(module, tmp_path)
    setattr(args, attribute, value)
    called = False

    def unexpected_import():
        nonlocal called
        called = True
        raise AssertionError("optional import ran before argument validation")

    monkeypatch.setattr(module, "require_calvin", unexpected_import)
    with pytest.raises(ValueError, match=error):
        module.run(args)
    assert called is False


def test_sharded_results_preserve_absolute_sequence_indices(tmp_path):
    module = importlib.import_module("eval_code.calvin_evaluate")
    result = module.SequenceResult(
        completed_subtasks=0,
        subtask_successes=(False,),
        actions=np.zeros((1, 7), dtype=np.float32),
        policy_latencies_ms=(1.0,),
        environment_latencies_ms=(2.0,),
        metrics={"executed_actions": 1},
    )

    module._write_results(
        tmp_path,
        {"sequences": [{"index": 12}]},
        [result],
        save_actions=True,
        sequence_start_index=12,
    )

    assert (tmp_path / "actions" / "sequence_000012.npz").is_file()
    assert not (tmp_path / "actions" / "sequence_000000.npz").exists()
    assert (tmp_path / "results_000012_000013.json").is_file()
    assert not (tmp_path / "results.json").exists()


def test_summary_reports_action_count_with_unambiguous_names():
    module = importlib.import_module("eval_code.calvin_evaluate")

    class AlwaysSuccessfulOracle:
        def get_task_info_for_set(self, start_info, current_info, task_filter):
            del start_info, current_info
            return task_filter

    summary, _ = module.evaluate_policy(
        env=_FakeEnvironment(),
        policy=_FakeStepPolicy(),
        task_oracle=AlwaysSuccessfulOracle(),
        sequences=[({}, ("first",)), ({}, ("first",))],
        annotations={"first": ["first"]},
        state_converter=lambda state: (state, state),
        max_steps=1,
        hf_cutoff_ratio=0.3,
    )

    assert summary["total_executed_actions"] == 2
    assert summary["mean_executed_actions_per_sequence"] == 1.0
    assert "executed_actions" not in summary["metrics"]


def test_calvin_launcher_requires_external_locations_and_contains_no_private_paths():
    launcher = Path(__file__).with_name("eval_calvin_chunkflow.sh").read_text(encoding="utf-8")

    assert ': "${CALVIN_ROOT:?set CALVIN_ROOT to the CALVIN checkout}"' in launcher
    assert ': "${CHUNKFLOW_CHECKPOINT:?set CHUNKFLOW_CHECKPOINT to a trained checkpoint}"' in launcher
    assert ': "${CALVIN_DATASET:?set CALVIN_DATASET to the CALVIN dataset root}"' in launcher
    lowered = launcher.lower()
    for forbidden in (
        "/" + "users/",
        "/" + "home/",
    ):
        assert forbidden not in lowered


def test_calvin_launcher_resolves_external_paths_before_changing_directory(tmp_path):
    launcher = Path(__file__).with_name("eval_calvin_chunkflow.sh").resolve()
    for directory in ("calvin", "dataset", "checkpoint"):
        (tmp_path / directory).mkdir()
    environment = os.environ.copy()
    environment.update(
        {
            "CALVIN_ROOT": "calvin",
            "CALVIN_DATASET": "dataset",
            "CHUNKFLOW_CHECKPOINT": "checkpoint",
            "CHUNKFLOW_CONFIG": "test_config",
            "PYTHON_BIN": "/bin/echo",
        }
    )

    completed = subprocess.run(
        ["bash", str(launcher)],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    arguments = shlex.split(completed.stdout)

    assert arguments[arguments.index("--calvin-root") + 1] == str((tmp_path / "calvin").resolve())
    assert arguments[arguments.index("--dataset-path") + 1] == str((tmp_path / "dataset").resolve())
    assert arguments[arguments.index("--checkpoint-dir") + 1] == str((tmp_path / "checkpoint").resolve())
