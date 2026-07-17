import numpy as np
import pytest

from openpi.models import model as _model
from openpi.policies import droid_policy
from openpi.policies import libero_policy
from openpi.policies import truth_policy_cartesian

CHUNKFLOW_FIELDS = {
    "action_history": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    "action_history_mask": np.array([True, False]),
    "rewards": np.array([0.0, 1.0], dtype=np.float32),
    "discounts": np.array([1.0, 0.0], dtype=np.float32),
    "executed_actions": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
}


def _libero_case():
    return libero_policy.make_libero_example(), libero_policy.LiberoInputs(_model.ModelType.PI05)


def _droid_case():
    return droid_policy.make_droid_example(), droid_policy.DroidInputs(_model.ModelType.PI05)


def _truth_cartesian_case():
    return (
        truth_policy_cartesian.make_droid_example(),
        truth_policy_cartesian.TruthInputsCartesian(_model.ModelType.PI05),
    )


def _truth_joint_case():
    data = truth_policy_cartesian.make_droid_example()
    data["observation/joint_position"] = np.arange(6, dtype=np.float32)
    return data, truth_policy_cartesian.TruthInputsJointWithoutGripper(_model.ModelType.PI05)


@pytest.mark.parametrize(
    "make_case",
    [_libero_case, _droid_case, _truth_cartesian_case, _truth_joint_case],
    ids=["libero", "droid", "truth-cartesian", "truth-joint"],
)
def test_policy_inputs_preserve_chunkflow_fields(make_case):
    data, adapter = make_case()
    data.update(CHUNKFLOW_FIELDS)

    transformed = adapter(data)

    for key, expected in CHUNKFLOW_FIELDS.items():
        np.testing.assert_array_equal(transformed[key], expected)


def test_libero_does_not_synthesize_chunk_terminal_from_episode_success():
    data = libero_policy.make_libero_example()
    data["actions"] = np.zeros((4, 7), dtype=np.float32)
    data["success"] = np.True_

    transformed = libero_policy.LiberoInputs(_model.ModelType.PI05)(data)

    assert "rewards" not in transformed
    assert "discounts" not in transformed


def test_libero_does_not_invent_discounts_when_only_rewards_are_present():
    data = libero_policy.make_libero_example()
    data["rewards"] = np.array([0.0, 1.0], dtype=np.float32)

    transformed = libero_policy.LiberoInputs(_model.ModelType.PI05)(data)

    np.testing.assert_array_equal(transformed["rewards"], data["rewards"])
    assert "discounts" not in transformed
