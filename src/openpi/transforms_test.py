import numpy as np
import pytest

import openpi.models.tokenizer as _tokenizer
from openpi.shared import normalize as _normalize
import openpi.transforms as _transforms


def test_repack_transform():
    transform = _transforms.RepackTransform(
        structure={
            "a": {"b": "b/c"},
            "d": "e/f",
        }
    )
    item = {"b": {"c": 1}, "e": {"f": 2}}
    assert transform(item) == {"a": {"b": 1}, "d": 2}


def test_delta_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.DeltaActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 2, 5], [5, 4, 7]]))


def test_delta_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.DeltaActions(mask=None)
    assert transform(item) is item

    # No-op when there are no action-like fields in the input.
    del item["actions"]
    transform = _transforms.DeltaActions(mask=[True, False])
    assert transform(item) is item


def test_chunkflow_action_fields_share_delta_normalization_and_padding():
    item = {
        "state": np.array([8.0], dtype=np.float32),
        "actions": np.array([[10.0], [12.0]], dtype=np.float32),
        "action_history": np.array([[9.0], [11.0]], dtype=np.float32),
        "executed_actions": np.array([[13.0]], dtype=np.float32),
    }
    originals = {key: value.copy() for key, value in item.items()}
    stats = {
        "actions": _normalize.NormStats(
            mean=np.array([10.0], dtype=np.float32),
            std=np.array([2.0], dtype=np.float32),
        )
    }

    transformed = _transforms.compose(
        [
            _transforms.DeltaActions(mask=[True]),
            _transforms.Normalize(stats),
            _transforms.PadStatesAndActions(model_action_dim=3),
        ]
    )(item)

    np.testing.assert_allclose(transformed["state"], [8.0, 0.0, 0.0])
    np.testing.assert_allclose(
        transformed["actions"],
        [[-4.0, 0.0, 0.0], [-3.0, 0.0, 0.0]],
        atol=1e-5,
    )
    np.testing.assert_allclose(
        transformed["action_history"],
        [[-4.5, 0.0, 0.0], [-3.5, 0.0, 0.0]],
        atol=1e-5,
    )
    np.testing.assert_allclose(
        transformed["executed_actions"],
        [[-2.5, 0.0, 0.0]],
        atol=1e-5,
    )
    for key, value in originals.items():
        np.testing.assert_array_equal(item[key], value)


def test_normalize_uses_action_quantiles_for_chunkflow_aliases():
    stats = {
        "actions": _normalize.NormStats(
            mean=np.array([0.0]),
            std=np.array([1.0]),
            q01=np.array([2.0]),
            q99=np.array([6.0]),
        )
    }
    item = {
        "actions": np.array([[2.0], [6.0]]),
        "action_history": np.array([[3.0], [5.0]]),
    }

    transformed = _transforms.Normalize(stats, use_quantiles=True, strict=True)(item)

    np.testing.assert_allclose(transformed["actions"], [[-1.0], [1.0]], atol=1e-5)
    np.testing.assert_allclose(transformed["action_history"], [[-0.5], [0.5]], atol=1e-5)


def test_masked_history_padding_stays_zero_through_action_transforms():
    stats = {
        "actions": _normalize.NormStats(
            mean=np.array([10.0], dtype=np.float32),
            std=np.array([2.0], dtype=np.float32),
        )
    }
    item = {
        "state": np.array([8.0], dtype=np.float32),
        "actions": np.array([[10.0]], dtype=np.float32),
        "action_history": np.array([[0.0], [9.0]], dtype=np.float32),
        "action_history_mask": np.array([False, True]),
    }

    transformed = _transforms.compose(
        [
            _transforms.DeltaActions(mask=[True]),
            _transforms.Normalize(stats),
        ]
    )(item)

    np.testing.assert_allclose(transformed["action_history"], [[0.0], [-4.5]], atol=1e-5)


def test_delta_cartesian_pose_transforms_all_chunkflow_action_fields_without_mutation():
    state = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 9.0])
    item = {
        "state": state,
        "actions": np.array([[2.0, 4.0, 6.0, 0.2, 0.4, 0.6, 8.0]]),
        "action_history": np.array([[3.0, 5.0, 7.0, 0.3, 0.5, 0.7, 7.0]]),
        "executed_actions": np.array([[4.0, 6.0, 8.0, 0.4, 0.6, 0.8, 6.0]]),
    }
    originals = {key: value.copy() for key, value in item.items()}

    transformed = _transforms.DeltaCartesianPose(mask=_transforms.make_bool_mask(6, -1))(item)

    np.testing.assert_allclose(transformed["actions"], [[1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 8.0]])
    np.testing.assert_allclose(
        transformed["action_history"], [[2.0, 3.0, 4.0, 0.2, 0.3, 0.4, 7.0]]
    )
    np.testing.assert_allclose(
        transformed["executed_actions"], [[3.0, 4.0, 5.0, 0.3, 0.4, 0.5, 6.0]]
    )
    for key, value in originals.items():
        np.testing.assert_array_equal(item[key], value)


def test_absolute_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.AbsoluteActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 6, 5], [5, 8, 7]]))


def test_absolute_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.AbsoluteActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.AbsoluteActions(mask=[True, False])
    assert transform(item) is item


def test_make_bool_mask():
    assert _transforms.make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
    assert _transforms.make_bool_mask(2, 0, 2) == (True, True, True, True)


def test_tokenize_prompt():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=12)
    transform = _transforms.TokenizePrompt(tokenizer)

    data = transform({"prompt": "Hello, world!"})

    tok_prompt, tok_mask = tokenizer.tokenize("Hello, world!")
    assert np.allclose(tok_prompt, data["tokenized_prompt"])
    assert np.allclose(tok_mask, data["tokenized_prompt_mask"])


def test_tokenize_no_prompt():
    transform = _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer())

    with pytest.raises(ValueError, match="Prompt is required"):
        transform({})


def test_transform_dict():
    # Rename and remove keys.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a/b": "a/c", "a/c": None}, input)
    assert output == {"a": {"c": 1}}

    # Raises and error since the renamed key conflicts with an existing key.
    with pytest.raises(ValueError, match="Key 'a/c' already exists in output"):
        _transforms.transform_dict({"a/b": "a/c"}, input)

    # Full match is required and so nothing will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a": None}, input)
    assert output == input

    # The regex matches the entire key and so the entire input will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a.+": None}, input)
    assert output == {}

    # Replace keys using backreferences. All leaves named 'c' are replaced with 'd'.
    input = {"a": {"b": 1, "c": 1}, "b": {"c": 2}}
    output = _transforms.transform_dict({"(.+)/c": r"\1/d"}, input)
    assert output == {"a": {"b": 1, "d": 1}, "b": {"d": 2}}


def test_extract_prompt_from_task():
    transform = _transforms.PromptFromLeRobotTask({1: "Hello, world!"})

    data = transform({"task_index": 1})
    assert data["prompt"] == "Hello, world!"

    with pytest.raises(ValueError, match="task_index=2 not found in task mapping"):
        transform({"task_index": 2})
