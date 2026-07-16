from __future__ import annotations

import ast
import copy
import json
import logging
from pathlib import Path
import statistics
import tempfile
import time
import types
import typing

import numpy as np
import pytest

from openpi.shared import metrics


ROOT = Path(__file__).resolve().parents[3]


def test_linear_trajectory_and_seam_metrics():
    actions = np.array([[0.0], [1.0], [2.0], [3.0]])

    assert metrics.compute_msd_delta(actions, 1) == pytest.approx(1.0)
    assert metrics.compute_msd_delta(actions, 2) == pytest.approx(0.0)
    seam = metrics.compute_seam_metrics(
        np.array([[0.0], [2.0]]),
        np.array([[1.0], [4.0]]),
    )
    assert seam["bjump"] == pytest.approx(1.5)
    assert seam["bratio"] == pytest.approx(0.5)


def test_tv_l1_is_mean_per_step_l1_norm():
    actions = np.array([[0.0, 0.0], [1.0, -2.0], [2.0, -4.0]])

    assert metrics.compute_tv_l1(actions) == pytest.approx(3.0)


def test_hf_ratio_uses_true_nyquist_for_odd_lengths_and_is_scale_invariant():
    samples = np.arange(5, dtype=np.float64)
    below_cutoff = np.sin(2.0 * np.pi * samples / 5.0)[:, None]
    assert metrics.compute_hf_ratio(below_cutoff, cutoff_ratio=0.5) == pytest.approx(0.0, abs=1e-12)

    alternating = np.array([1.0, -1.0] * 4, dtype=np.float64)[:, None]
    reference = metrics.compute_hf_ratio(alternating, cutoff_ratio=0.5)
    assert reference == pytest.approx(1.0)
    for scale in (1e-200, 1e-8, 1e200):
        scaled = metrics.compute_hf_ratio(alternating * scale, cutoff_ratio=0.5)
        assert scaled == pytest.approx(reference)


def test_static_new_head_with_nonzero_seam_has_infinite_bratio():
    seam = metrics.compute_seam_metrics(
        np.array([[1.0], [1.0]]),
        np.array([[0.0], [0.0]]),
    )

    assert seam["bjump"] == pytest.approx(1.0)
    assert np.isinf(seam["bratio"])


def test_single_action_overlap_uses_zero_denominator_rule():
    mismatch = metrics.compute_seam_metrics(np.array([[1.0]]), np.array([[0.0]]))
    identical = metrics.compute_seam_metrics(np.array([[1.0]]), np.array([[1.0]]))

    assert mismatch["bjump"] == pytest.approx(1.0)
    assert np.isinf(mismatch["bratio"])
    assert identical == {"bjump": 0.0, "bratio": 0.0}


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_seam_metrics_are_stable_across_extreme_scales(scale):
    seam = metrics.compute_seam_metrics(
        np.array([[0.0], [2.0]]) * scale,
        np.array([[1.0], [4.0]]) * scale,
    )

    assert seam["bjump"] == pytest.approx(1.5 * scale, rel=1e-12, abs=0.0)
    assert seam["bratio"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "actions",
    [
        np.array([1.0, 2.0]),
        np.empty((2, 0)),
        np.array([[np.nan]]),
        np.array([[np.inf]]),
        np.array([[1.0 + 2.0j]]),
        np.array([["1.0"]]),
    ],
)
@pytest.mark.parametrize(
    "metric_fn",
    [metrics.compute_msd_delta, metrics.compute_tv_l1, metrics.compute_hf_ratio],
)
def test_action_metrics_reject_invalid_matrices(metric_fn, actions):
    with pytest.raises(ValueError, match="actions"):
        metric_fn(actions)


def test_seam_metrics_require_two_dimensional_finite_real_actions():
    with pytest.raises(ValueError, match="previous_tail"):
        metrics.compute_seam_metrics(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="new_head"):
        metrics.compute_seam_metrics(np.ones((2, 1)), np.array([[np.inf], [0.0]]))


@pytest.mark.parametrize("cutoff_ratio", [-0.1, 1.1, np.nan, np.inf])
def test_frequency_cutoff_is_validated(cutoff_ratio):
    with pytest.raises(ValueError, match="cutoff_ratio"):
        metrics.compute_hf_ratio(np.ones((8, 2)), cutoff_ratio=cutoff_ratio)


def test_reasoning_latency_is_total_time_per_executed_action():
    assert metrics.compute_average_reasoning_latency([10.0, 20.0], [2, 3]) == pytest.approx(6.0)


def test_reasoning_latency_preserves_large_python_integer_counts():
    count = 2**64 + 1

    assert metrics.compute_average_reasoning_latency([10.0], [count]) == pytest.approx(10.0 / count)


@pytest.mark.parametrize("latencies", [[True], ["10.0"], [np.nan], [np.inf], [-1.0]])
def test_reasoning_latency_rejects_non_real_or_invalid_latencies(latencies):
    with pytest.raises(ValueError, match="latencies"):
        metrics.compute_average_reasoning_latency(latencies, [1])


@pytest.mark.parametrize("counts", [[True], [1.5], ["1"]])
def test_reasoning_latency_rejects_non_integral_counts(counts):
    with pytest.raises(ValueError, match="action counts"):
        metrics.compute_average_reasoning_latency([10.0], counts)


@pytest.mark.parametrize(
    ("latencies", "counts"),
    [
        ([10.0], [0]),
        ([10.0], [-1]),
        ([10.0, 20.0], [2]),
        ([[10.0]], [1]),
    ],
)
def test_reasoning_latency_rejects_invalid_action_counts(latencies, counts):
    with pytest.raises(ValueError, match="action count|equal-length vectors"):
        metrics.compute_average_reasoning_latency(latencies, counts)


def _method_calls(source: str, class_name: str | None, method_name: str) -> set[str]:
    tree = ast.parse(source)
    body = tree.body
    if class_name is not None:
        class_node = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        body = class_node.body
    method = next(
        node
        for node in body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name
    )
    return {
        f"{node.func.value.id}.{node.func.attr}"
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
    }


def _load_method(source: str, class_name: str, method_name: str, namespace: dict[str, object]):
    tree = ast.parse(source)
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name
    )
    method = copy.deepcopy(method)
    method.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    execution_namespace = dict(namespace)
    exec(compile(module, f"<{class_name}.{method_name}>", "exec"), execution_namespace)
    return execution_namespace[method_name]


def test_libero_evaluator_uses_canonical_metric_helpers():
    source = (ROOT / "eval_code/pi05_rtc_libero_eval.py").read_text(encoding="utf-8")

    assert "_metrics.compute_seam_metrics" in _method_calls(
        source, "TemporalMetrics", "compute_boundary_consistency"
    )
    assert "_metrics.compute_hf_ratio" in _method_calls(
        source, "TemporalMetrics", "compute_frequency_regularity"
    )
    global_calls = _method_calls(source, "TemporalMetrics", "compute_global_variation")
    assert {
        "_metrics.compute_tv_l1",
        "_metrics.compute_msd_delta",
    } <= global_calls
    assert "_metrics.compute_average_reasoning_latency" in _method_calls(
        source, "Pi05RTCEvaluator", "_run_episode"
    )


def test_libero_temporal_metric_wrappers_execute_canonical_results():
    source = (ROOT / "eval_code/pi05_rtc_libero_eval.py").read_text(encoding="utf-8")
    namespace = {
        "Dict": typing.Dict,
        "List": typing.List,
        "_metrics": metrics,
        "np": np,
    }
    boundary = _load_method(source, "TemporalMetrics", "compute_boundary_consistency", namespace)
    frequency = _load_method(source, "TemporalMetrics", "compute_frequency_regularity", namespace)
    variation = _load_method(source, "TemporalMetrics", "compute_global_variation", namespace)

    transitions = [
        {
            "prev_tail": np.array([[0.0], [2.0]]),
            "new_head": np.array([[1.0], [4.0]]),
        }
    ]
    seam = boundary(transitions)
    assert seam["bjump"] == pytest.approx(1.5)
    assert seam["bratio"] == pytest.approx(0.5)

    huge = float(np.finfo(np.float32).max) * 2.0
    large_finite = boundary(
        [{"prev_tail": np.array([[huge], [huge]]), "new_head": np.array([[huge], [huge]])}]
    )
    assert large_finite["bjump"] == pytest.approx(0.0)

    samples = np.arange(5, dtype=np.float64)
    low_frequency = np.sin(2.0 * np.pi * samples / 5.0)[:, None]
    assert frequency(low_frequency, fc_ratio=0.5)["hf_ratio"] == pytest.approx(0.0, abs=1e-12)

    actions = np.array([[0.0, 0.0], [1.0, -2.0], [2.0, -4.0]])
    wrapped = variation(actions)
    assert wrapped["tv_l1"] == pytest.approx(3.0)
    assert wrapped["msd_d1"] == pytest.approx(5.0)


def test_libero_episode_counts_only_successfully_executed_actions():
    source = (ROOT / "eval_code/pi05_rtc_libero_eval.py").read_text(encoding="utf-8")

    class FakePolicy:
        def reset(self):
            pass

        def infer(self, _observation):
            return {"actions": np.array([0.0], dtype=np.float32)}

        def get_metrics(self):
            return {"chunk_transitions": []}

    class FailingOnSecondStepEnv:
        def __init__(self):
            self.calls = 0

        def reset(self):
            pass

        def set_init_state(self, _state):
            return {}

        def step(self, _action):
            self.calls += 1
            if self.calls == 1:
                return {}, 0.0, False, {}
            raise RuntimeError("execution failed")

        def close(self):
            pass

    class FakeTemporalMetrics:
        @staticmethod
        def compute_boundary_consistency(_transitions):
            return {}

        @staticmethod
        def compute_frequency_regularity(_actions, fc_ratio):
            del fc_ratio
            return {}

        @staticmethod
        def compute_global_variation(_actions):
            return {}

    env = FailingOnSecondStepEnv()
    with tempfile.TemporaryDirectory() as temporary_directory:
        evaluator = types.SimpleNamespace(
            args=types.SimpleNamespace(
                fc_ratio=0.25,
                blending_method="linear",
                num_steps_wait=0,
                overlap_size=1,
                replan_interval=1,
                save_pred_actions=False,
                save_videos=False,
            ),
            data_dir=Path(temporary_directory),
            max_steps=2,
            policy=FakePolicy(),
        )
        evaluator._get_libero_env = lambda _task: (env, "task")
        evaluator._prepare_observation = lambda _obs, _description: ({}, None, None)

        run_episode = _load_method(
            source,
            "Pi05RTCEvaluator",
            "_run_episode",
            {
                "Any": typing.Any,
                "Dict": typing.Dict,
                "List": typing.List,
                "TemporalMetrics": FakeTemporalMetrics,
                "Tuple": typing.Tuple,
                "_metrics": metrics,
                "json": json,
                "logging": logging,
                "np": np,
                "statistics": statistics,
                "time": time,
            },
        )
        _success, episode = run_episode(
            evaluator,
            task_id=0,
            episode_idx=0,
            task=object(),
            task_description="task",
            initial_states=[object()],
        )

    assert episode["chunk_action_counts"] == [1, 0]
    assert episode["num_chunks"] == 2
    assert episode["arl"] == pytest.approx(sum(episode["chunk_inference_times"]))
