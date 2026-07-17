from collections.abc import Callable, Mapping, Sequence
import dataclasses
import re
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats


T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item[k], self.structure)


@dataclasses.dataclass(frozen=True)
class RepackTransformOptional(DataTransformFn):
    """Like RepackTransform, but safely skips optional leaves if the source key is missing.

    This is useful when some dataset fields (e.g., rewards/discounts) may not be present for
    every sample. Required leaves will still raise if missing.
    """

    structure: at.PyTree[str]
    # Keys to treat as optional (leaf names in the output structure).
    optional_leaf_names: tuple[str, ...] = ("rewards", "discounts")

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)

        def build(node, out_name: str | None = None):
            # Returns (value, present: bool)
            if isinstance(node, dict):
                result = {}
                any_present = False
                for k, v in node.items():
                    val, present = build(v, k)
                    if present:
                        result[k] = val
                        any_present = True
                    else:
                        # If this child is missing and is required (not in optional), raise.
                        if k not in self.optional_leaf_names:
                            raise KeyError(f"Required key '{k}' missing in input when repacking.")
                return result, any_present
            else:
                # Leaf: node is a flattened input path string
                if node in flat_item:
                    return flat_item[node], True
                # Missing leaf: allow skip only if optional
                if out_name in self.optional_leaf_names:
                    return None, False
                raise KeyError(f"Required key '{out_name}' (path '{node}') missing in input when repacking.")

        result, _ = build(self.structure, None)
        return result

@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False
    # Action-valued fields that share the canonical ``actions`` statistics.
    action_aliases: tuple[str, ...] = ("action_history", "executed_actions")

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        normalize_fn = self._normalize_quantile if self.use_quantiles else self._normalize
        normalized = apply_tree(
            data,
            self.norm_stats,
            normalize_fn,
            strict=self.strict,
        )

        flat_stats = flatten_dict(self.norm_stats)
        action_stats = flat_stats.get("actions")
        flat_data = flatten_dict(normalized)
        if action_stats is not None:
            for alias in self.action_aliases:
                # An explicitly configured alias has already been normalized above.
                if alias in flat_data and flat_data[alias] is not None and alias not in flat_stats:
                    flat_data[alias] = normalize_fn(flat_data[alias], action_stats)
        if (
            flat_data.get("action_history") is not None
            and flat_data.get("action_history_mask") is not None
        ):
            flat_data["action_history"] = _zero_masked_history(
                flat_data["action_history"], flat_data["action_history_mask"]
            )
        return unflatten_dict(flat_data)

    def _normalize(self, x, stats: NormStats):
        mean, std = stats.mean[..., : x.shape[-1]
                               ], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        data["image"] = {k: image_tools.resize_with_pad(
            v, self.height, self.width) for k, v in data["image"].items()}
        return data


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None
    action_keys: tuple[str, ...] = ("actions", "action_history", "executed_actions")

    def __call__(self, data: DataDict) -> DataDict:
        present_keys = tuple(key for key in self.action_keys if key in data and data[key] is not None)
        if not present_keys or self.mask is None:
            return data

        state = np.asarray(data["state"])
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        if state.shape[-1] < dims:
            raise ValueError(f"state has {state.shape[-1]} dimensions but delta mask has {dims}")
        offset = np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)

        transformed = dict(data)
        for key in present_keys:
            actions = np.array(data[key], copy=True)
            if actions.ndim < 2:
                raise ValueError(f"{key} must have an action-sequence axis")
            if actions.shape[-1] < dims:
                raise ValueError(f"{key} has {actions.shape[-1]} dimensions but delta mask has {dims}")
            actions[..., :dims] = actions[..., :dims] - offset
            if key == "action_history" and "action_history_mask" in data:
                actions = _zero_masked_history(actions, data["action_history_mask"])
            transformed[key] = actions
        return transformed


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(
            np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        # actions = np.array(actions, copy=True)
        # offset = np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        # actions[..., :dims] = actions[..., :dims] + offset
        # data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(
            prompt, state, actions)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        # Model outputs are saved in "actions", but for FAST models they represent tokens.
        tokens = data.pop("actions")
        actions = self.tokenizer.extract_actions(tokens.astype(
            np.int32), self.action_horizon, self.action_dim)
        return {
            **data,
            "actions": actions,
        }


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(
                f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int
    action_keys: tuple[str, ...] = ("actions", "action_history", "executed_actions")

    def __call__(self, data: DataDict) -> DataDict:
        transformed = dict(data)
        transformed["state"] = pad_to_dim(
            np.array(data["state"], copy=True), self.model_action_dim, axis=-1
        )
        for key in self.action_keys:
            if key in data and data[key] is not None:
                transformed[key] = pad_to_dim(
                    np.array(data[key], copy=True), self.model_action_dim, axis=-1
                )
        return transformed


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(
                    repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i: i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def _zero_masked_history(history: np.ndarray, mask: np.ndarray) -> np.ndarray:
    history = np.asarray(history)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != history.shape[:-1]:
        raise ValueError(
            f"action_history_mask shape {mask.shape} must match action_history prefix {history.shape[:-1]}"
        )
    return np.where(mask[..., None], history, np.zeros((), dtype=history.dtype))


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )


@dataclasses.dataclass(frozen=True)
class DeltaCartesianPose(DataTransformFn):
    """将绝对位姿动作转换为增量位姿动作空间。

    前3维是位置(x,y,z)，后3维是欧拉角(roll,pitch,yaw)。
    位置使用欧几里得距离，角度使用角度差。
    """

    # 布尔掩码，用于指定要转换为增量动作空间的位姿维度
    # 长度可以小于实际维度数。如果为None，此变换无效。
    # 参见 `make_bool_mask` 了解更多详情。
    mask: Sequence[bool] | None
    action_keys: tuple[str, ...] = ("actions", "action_history", "executed_actions")

    def __call__(self, data: DataDict) -> DataDict:
        present_keys = tuple(key for key in self.action_keys if key in data and data[key] is not None)
        if not present_keys or self.mask is None:
            return data

        state = np.asarray(data["state"])
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]

        if dims not in (3, 7):
            raise ValueError("DeltaCartesianPose mask must describe 3 position or 7 pose/gripper dimensions")
        if state.shape[-1] < dims:
            raise ValueError(f"state has {state.shape[-1]} dimensions but delta mask has {dims}")

        transformed = dict(data)
        for key in present_keys:
            actions = np.array(data[key], copy=True)
            if actions.ndim < 2 or actions.shape[-1] < dims:
                raise ValueError(f"{key} must have shape [..., horizon, action_dim >= {dims}]")

            pos_offset = np.expand_dims(np.where(mask[:3], state[..., :3], 0), axis=-2)
            actions[..., :3] = actions[..., :3] - pos_offset
            if dims == 7:
                angle_offset = np.expand_dims(np.where(mask[3:6], state[..., 3:6], 0), axis=-2)
                angles = actions[..., 3:6] - angle_offset
                actions[..., 3:6] = (angles + np.pi) % (2 * np.pi) - np.pi
            if key == "action_history" and "action_history_mask" in data:
                actions = _zero_masked_history(actions, data["action_history_mask"])
            transformed[key] = actions
        return transformed


@dataclasses.dataclass(frozen=True)
class AbsoluteCartesianPose(DataTransformFn):
    """将增量位姿动作转换为绝对位姿动作空间。

    前3维是位置(x,y,z)，后3维是欧拉角(roll,pitch,yaw)。
    位置使用欧几里得距离，角度使用角度和。
    """

    # 布尔掩码，用于指定要转换为绝对动作空间的位姿维度
    # 长度可以小于实际维度数。如果为None，此变换无效。
    # 参见 `make_bool_mask` 了解更多详情。
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]

        if dims < 3:
            raise ValueError(
                "AbsoluteCartesianPose requires at least 3 dimensions (position)")

        # 位置部分（前3维）：按掩码相加
        pos_mask = mask[:3]
        pos_state = state[..., :3]
        pos_state_masked = np.where(pos_mask, pos_state, 0)
        actions[..., :3] += np.expand_dims(pos_state_masked, axis=-2)

        # 姿态欧拉角部分（后3维，弧度）：仅当提供6维时处理
        if dims == 7:
            ang_mask = mask[3:6]
            ang_state = state[..., 3:6]
            ang_state_masked = np.where(ang_mask, ang_state, 0)
            ang = actions[..., 3:6] + np.expand_dims(ang_state_masked, axis=-2)
            # 归一化到[-pi, pi]
            ang = (ang + np.pi) % (2 * np.pi) - np.pi
            actions[..., 3:6] = ang
        elif 3 < dims < 7:
            raise ValueError(
                "AbsoluteCartesianPose dims must be 3 or 6 (got partial angles)")
        data["actions"] = actions

        return data
