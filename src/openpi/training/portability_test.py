from __future__ import annotations

import ast
import importlib.util
import json
import os
import pathlib
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
import types
import unittest
from unittest import mock
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[3]

TASK_3B_RUNTIME_PATHS = (
    ".gitignore",
    "src/openpi/training/config.py",
    "src/openpi/training/data_loader.py",
    "src/openpi/training/truth_rlds_dataset.py",
    "scripts/generate_libero_success_map.py",
    "scripts/docker/compose.yml",
    "download_pi0.py",
    "examples/aloha_real/robot_utils.py",
    "examples/convert_jax_model_to_pytorch.py",
)

TASK_3C_EVALUATOR_PATHS = (
    "eval_code/eval_action_smoothness.py",
    "eval_code/pi05_rtc_libero_eval.py",
    "eval_code/run_pi05_rtc_eval.sh",
    "eval_code/real_robot_offline_eval.py",
    "eval_code/real_robot_dataset_eval.py",
)

FORBIDDEN_MARKERS = (
    "/" + "Users/",
    "/" + "home/",
    "/" + "mnt/",
    "tos" + "://",
    "yao" + "mingyuan",
    "lucy" + "shi",
    "openpi_" + "main",
)
FORBIDDEN_DATASET_RE = re.compile(r"\bcyto" + r"derm\d*_dataset")

OLD_EVALUATORS = (
    "eval_code/evaluate_truth_cartesion_position_action_add_step.py",
    "eval_code/evaluate_truth_dataset_rlds_plot_every_cartesion_add_step.py",
)
NEW_EVALUATORS = (
    "eval_code/real_robot_offline_eval.py",
    "eval_code/real_robot_dataset_eval.py",
)

EXPECTED_EVALUATOR_DEFAULTS = {
    "eval_code/real_robot_offline_eval.py": {
        "dataset_path": "datasets/real_robot",
        "dataset_name": "chunkflow_real_wood",
        "config": "pi05_truth_finetune_cartesian_wood",
        "checkpoint_dir": "",
        "output_dir": "outputs/real_robot/offline_eval",
    },
    "eval_code/real_robot_dataset_eval.py": {
        "dataset_path": "datasets/real_robot",
        "dataset_name": "chunkflow_real_cloth_v2",
        "config": "pi05_truth_finetune_cartesian_cloth_new",
        "checkpoint_dir": "",
        "output_dir": "outputs/real_robot/dataset_eval",
    },
    "eval_code/pi05_rtc_libero_eval.py": {
        "checkpoint_dir": "",
        "output_dir": "outputs/libero/pi05_rtc_eval",
    },
}

EXPECTED_ENV_DEFAULTS = {
    "CHUNKFLOW_LIBERO_DATASET": "datasets/libero_lerobot",
    "CHUNKFLOW_REAL_DATA_DIR": "datasets/real_robot",
    "CHUNKFLOW_PI05_BASE_CHECKPOINT": "gs://openpi-assets/checkpoints/pi05_base/params",
    "CHUNKFLOW_PAPER_REPO_ID": "chunkflow/libero",
    "CHUNKFLOW_PAPER_DATASET_ROOT": "datasets/chunkflow_lerobot",
    "CHUNKFLOW_SUPERVISED_CHECKPOINT": "checkpoints/pi05_chunkflow_paper_bc/chunkflow_bc/29999/params",
    "CHUNKFLOW_REAL_JOINT_STAGE1_CHECKPOINT": "checkpoints/chunkflow_real_joint_stage1/params",
    "CHUNKFLOW_REAL_JOINT_STAGE2_CHECKPOINT": "checkpoints/chunkflow_real_joint_stage2/params",
}

EXPECTED_DERIVED_ENV_DEFAULTS = {
    "CHUNKFLOW_SUPERVISED_ASSETS": (
        'os.environ.get("CHUNKFLOW_SUPERVISED_ASSETS", '
        'f"{CHUNKFLOW_SUPERVISED_CHECKPOINT.rsplit(\'/\', 1)[0]}/assets")'
    ),
}

EXPECTED_REAL_CONFIGS = {
    "pi05_truth_finetune_cartesian_wood": (
        "RLDSTruthCartesianDataConfig",
        "chunkflow_real_wood",
        "truth_rlds_dataset.TruthActionSpace.CARTESIAN_POSITION",
        False,
        "CHUNKFLOW_PI05_BASE_CHECKPOINT",
    ),
    "pi05_truth_finetune_cartesian_cloth_downsampled": (
        "RLDSTruthCartesianDataConfig",
        "chunkflow_real_cloth",
        "truth_rlds_dataset.TruthActionSpace.CARTESIAN_POSITION",
        True,
        "CHUNKFLOW_PI05_BASE_CHECKPOINT",
    ),
    "pi05_truth_finetune_cartesian_cloth_new": (
        "RLDSTruthCartesianDataConfig",
        "chunkflow_real_cloth_v2",
        "truth_rlds_dataset.TruthActionSpace.CARTESIAN_POSITION",
        False,
        "CHUNKFLOW_PI05_BASE_CHECKPOINT",
    ),
    "pi05_truth_finetune_cartesian_catch_wood_and_move": (
        "RLDSTruthCartesianDataConfig",
        "chunkflow_real_block",
        "truth_rlds_dataset.TruthActionSpace.CARTESIAN_POSITION",
        False,
        "CHUNKFLOW_PI05_BASE_CHECKPOINT",
    ),
    "pi05_cytoderm11_joint_arm_move_chunkflow": (
        "RLDSTruthJointWithoutGripperDataConfig",
        "chunkflow_real_joint",
        "truth_rlds_dataset.TruthActionSpace.JOINT_POSITION",
        False,
        "CHUNKFLOW_REAL_JOINT_STAGE1_CHECKPOINT",
    ),
    "pi05_cytoderm13_joint_arm_move_chunkflow": (
        "RLDSTruthJointWithoutGripperDataConfig",
        "chunkflow_real_joint_v2",
        "truth_rlds_dataset.TruthActionSpace.JOINT_POSITION",
        False,
        "CHUNKFLOW_REAL_JOINT_STAGE2_CHECKPOINT",
    ),
}

EXPECTED_DOCKER_MOUNTS = {
    "CHUNKFLOW_SOURCE_DIR": "/app",
    "OPENPI_DATA_HOME": "/openpi_assets",
    "CHUNKFLOW_DATA_DIR": "/app/datasets",
    "CHUNKFLOW_CHECKPOINT_DIR": "/app/checkpoints",
    "CHUNKFLOW_OUTPUT_DIR": "/app/outputs",
}

LIBERO_COMMIT = "f78abd68ee283de9f9be3c8f7e2a9ad60246e95c"


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _args_default(source: str, field_name: str) -> object:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Args":
            for statement in node.body:
                if (
                    isinstance(statement, ast.AnnAssign)
                    and isinstance(statement.target, ast.Name)
                    and statement.target.id == field_name
                    and statement.value is not None
                ):
                    return ast.literal_eval(statement.value)
    raise AssertionError(f"Args.{field_name} default not found")


def _environment_defaults(source: str) -> dict[str, str]:
    defaults: dict[str, str] = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        value = node.value
        if not isinstance(target, ast.Name) or not isinstance(value, ast.Call):
            continue
        if len(value.args) < 2:
            continue
        try:
            variable = ast.literal_eval(value.args[0])
            default = ast.literal_eval(value.args[1])
        except (ValueError, TypeError):
            continue
        if variable == target.id and isinstance(default, str):
            defaults[target.id] = default
    return defaults


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    return next((keyword.value for keyword in call.keywords if keyword.arg == name), None)


def _train_config_calls(source: str) -> dict[str, ast.Call]:
    configs: dict[str, ast.Call] = {}
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or _call_name(node.func) != "TrainConfig":
            continue
        name_node = _keyword(node, "name")
        if isinstance(name_node, ast.Constant) and isinstance(name_node.value, str):
            configs[name_node.value] = node
    return configs


def _argparse_default_node(source: str, option: str) -> ast.expr:
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or _call_name(node.func) != "parser.add_argument":
            continue
        options: list[object] = []
        for argument in node.args:
            try:
                options.append(ast.literal_eval(argument))
            except (ValueError, TypeError):
                pass
        if option not in options:
            continue
        default = _keyword(node, "default")
        if default is None:
            raise AssertionError(f"{option} default not found")
        return default
    raise AssertionError(f"{option} argument not found")


def _argparse_default(source: str, option: str) -> object:
    return ast.literal_eval(_argparse_default_node(source, option))


def _top_level_function(source: str, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    return next(
        (
            node
            for node in ast.parse(source).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
        ),
        None,
    )


def _load_isolated_function(
    source: str,
    name: str,
    namespace: dict[str, object] | None = None,
):
    function = _top_level_function(source, name)
    if function is None:
        raise AssertionError(f"{name} function not found")
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    execution_namespace = {} if namespace is None else dict(namespace)
    exec(compile(module, f"<{name}>", "exec"), execution_namespace)
    return execution_namespace[name]


def _load_module_with_stubs(
    relative_path: str,
    module_name: str,
    stubs: dict[str, types.ModuleType],
):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {relative_path}")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


def _package_stub(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _load_truth_dataset_module():
    # Load NumPy before patch.dict restores sys.modules, avoiding repeated imports.
    importlib.import_module("numpy")
    openpi = _package_stub("openpi")
    # Keep pure local training helpers importable while replacing download I/O.
    openpi.__path__ = [str(ROOT / "src/openpi")]
    shared = _package_stub("openpi.shared")
    download = types.ModuleType("openpi.shared.download")
    openpi.shared = shared
    shared.download = download

    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda iterable, **_: iterable

    return _load_module_with_stubs(
        "src/openpi/training/truth_rlds_dataset.py",
        "task_3b_truth_rlds_dataset",
        {
            "openpi": openpi,
            "openpi.shared": shared,
            "openpi.shared.download": download,
            "tqdm": tqdm_module,
        },
    )


def _contains_call(node: ast.AST, function_name: str) -> bool:
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        function = child.func
        if isinstance(function, ast.Name) and function.id == function_name:
            return True
    return False


def _checkpoint_validation_precedes_evaluator(source: str, evaluator_name: str) -> bool:
    tree = ast.parse(source)
    validator = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_require_checkpoint_dir"
        ),
        None,
    )
    if validator is None:
        return False
    has_required_error = any(
        isinstance(node, ast.Raise)
        and isinstance(node.exc, ast.Call)
        and isinstance(node.exc.func, ast.Name)
        and node.exc.func.id == "ValueError"
        and any(
            isinstance(argument, ast.Constant)
            and argument.value == "--checkpoint-dir is required"
            for argument in node.exc.args
        )
        for node in ast.walk(validator)
    )
    if not has_required_error:
        return False
    main = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "main"
        ),
        None,
    )
    if main is None:
        return False
    validation_steps = [
        index for index, statement in enumerate(main.body) if _contains_call(statement, "_require_checkpoint_dir")
    ]
    evaluator_steps = [
        index for index, statement in enumerate(main.body) if _contains_call(statement, evaluator_name)
    ]
    return bool(validation_steps and evaluator_steps and min(validation_steps) < min(evaluator_steps))


class PortabilityTest(unittest.TestCase):
    def test_project_metadata_and_dependency_sources_are_portable(self) -> None:
        pyproject = tomllib.loads(_read("pyproject.toml"))
        project = pyproject["project"]
        sources = pyproject["tool"]["uv"]["sources"]
        failures: list[str] = []

        if project.get("name") != "chunkflow-openpi":
            failures.append(f"project name is {project.get('name')!r}")
        if project.get("description") != "ChunkFlow: continuity-consistent chunked policy learning on openpi/π0.5":
            failures.append(f"project description is {project.get('description')!r}")
        expected_urls = {
            "Repository": "https://github.com/yangzhaojason/ChunkFlow",
            "Project": "https://cytoderm-ai.github.io/",
        }
        if project.get("urls") != expected_urls:
            failures.append(f"project URLs are {project.get('urls')!r}")
        wheel_packages = (
            pyproject.get("tool", {})
            .get("hatch", {})
            .get("build", {})
            .get("targets", {})
            .get("wheel", {})
            .get("packages")
        )
        if wheel_packages != ["src/openpi"]:
            failures.append(f"Hatch wheel packages are {wheel_packages!r}")

        expected_sources = {
            "openpi-client": {"workspace": True},
            "lerobot": {
                "git": "https://github.com/huggingface/lerobot",
                "rev": "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5",
            },
            "libero": {
                "git": "https://github.com/Lifelong-Robot-Learning/LIBERO",
                "rev": LIBERO_COMMIT,
            },
            "dlimp": {
                "git": "https://github.com/kvablack/dlimp",
                "rev": "ad72ce3a9b414db2185bc0b38461d4101a65477a",
            },
        }
        for name, expected in expected_sources.items():
            if sources.get(name) != expected:
                failures.append(f"source {name} is {sources.get(name)!r}")
        for name, source in sources.items():
            local_path = source.get("path")
            if local_path is not None and Path(local_path).is_absolute():
                failures.append(f"source {name} uses absolute path {local_path!r}")

        lock = tomllib.loads(_read("uv.lock"))
        locked_sources = {
            package["name"]: package.get("source", {})
            for package in lock.get("package", [])
            if package.get("name") in {"lerobot", "libero"}
        }
        for name, source in locked_sources.items():
            directory = source.get("directory")
            if directory is not None and Path(directory).is_absolute():
                failures.append(f"locked source {name} uses absolute directory {directory!r}")
        locked_lerobot = locked_sources.get("lerobot", {}).get("git", "")
        if not locked_lerobot.endswith("#0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"):
            failures.append(f"locked lerobot source is {locked_lerobot!r}")
        locked_libero = locked_sources.get("libero", {}).get("git", "")
        locked_libero_commit = urlsplit(locked_libero).fragment
        if (
            "github.com/Lifelong-Robot-Learning/LIBERO" not in locked_libero
            or re.fullmatch(r"[0-9a-f]{40}", locked_libero_commit) is None
            or locked_libero_commit != LIBERO_COMMIT
        ):
            failures.append(f"locked libero source is {locked_libero!r}")

        if failures:
            self.fail("\n" + "\n".join(failures))

    def test_task_3b_runtime_files_have_no_internal_defaults(self) -> None:
        failures: list[str] = []
        for relative_path in TASK_3B_RUNTIME_PATHS:
            path = ROOT / relative_path
            if not path.is_file():
                failures.append(f"missing runtime file: {relative_path}")
                continue
            text = path.read_text(encoding="utf-8")
            for marker in FORBIDDEN_MARKERS:
                if marker in text:
                    failures.append(f"{relative_path}: contains {marker!r}")
            if FORBIDDEN_DATASET_RE.search(text):
                failures.append(f"{relative_path}: contains organization dataset id")

        if failures:
            self.fail("\n" + "\n".join(failures))

    def test_task_3c_evaluator_files_have_no_internal_defaults(self) -> None:
        failures: list[str] = []
        for relative_path in TASK_3C_EVALUATOR_PATHS:
            path = ROOT / relative_path
            if not path.is_file():
                failures.append(f"missing evaluator file: {relative_path}")
                continue
            text = path.read_text(encoding="utf-8")
            for marker in FORBIDDEN_MARKERS:
                if marker in text:
                    failures.append(f"{relative_path}: contains {marker!r}")
            if FORBIDDEN_DATASET_RE.search(text):
                failures.append(f"{relative_path}: contains organization dataset id")

        if failures:
            self.fail("\n" + "\n".join(failures))

    def test_task_3b_gitignore_has_no_organization_dataset_entries(self) -> None:
        if FORBIDDEN_DATASET_RE.search(_read(".gitignore")):
            self.fail(".gitignore retains organization dataset entries")

    def test_task_3c_curated_evaluator_layout_and_scanner_have_no_exemptions(self) -> None:
        failures: list[str] = []
        if (ROOT / ".gitmodules").exists():
            failures.append(".gitmodules still exists")
        for relative_path in OLD_EVALUATORS:
            if (ROOT / relative_path).exists():
                failures.append(f"old evaluator still exists: {relative_path}")
        for relative_path in NEW_EVALUATORS:
            if not (ROOT / relative_path).is_file():
                failures.append(f"new evaluator is missing: {relative_path}")

        scanner = _read("scripts/audit_release.py")
        if "INTERNAL_PATH_EXEMPTIONS" in scanner:
            failures.append("release scanner retains internal-path exemptions")

        if failures:
            self.fail("\n" + "\n".join(failures))

    def test_task_3b_environment_constants_and_retained_configs(self) -> None:
        failures: list[str] = []
        config_source = _read("src/openpi/training/config.py")
        tree = ast.parse(config_source)

        imports_os = any(
            isinstance(node, ast.Import) and any(alias.name == "os" for alias in node.names)
            for node in tree.body
        )
        if not imports_os:
            failures.append("config.py does not import os")

        environment_assignments = {
            node.targets[0].id: node.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id.startswith("CHUNKFLOW_")
        }
        expected_environment_names = set(EXPECTED_ENV_DEFAULTS) | set(EXPECTED_DERIVED_ENV_DEFAULTS)
        if set(environment_assignments) != expected_environment_names:
            failures.append(
                f"environment constants are {sorted(environment_assignments)!r}"
            )
        for name, default in EXPECTED_ENV_DEFAULTS.items():
            value = environment_assignments.get(name)
            if not isinstance(value, ast.Call):
                continue
            try:
                arguments = [ast.literal_eval(argument) for argument in value.args]
            except (ValueError, TypeError):
                arguments = []
            if (
                _call_name(value.func) != "os.environ.get"
                or arguments != [name, default]
                or value.keywords
            ):
                failures.append(f"{name} is defined as {ast.unparse(value)!r}")
        for name, expected_expression in EXPECTED_DERIVED_ENV_DEFAULTS.items():
            value = environment_assignments.get(name)
            expected_value = ast.parse(expected_expression, mode="eval").body
            if value is None or ast.dump(value, include_attributes=False) != ast.dump(
                expected_value, include_attributes=False
            ):
                rendered = ast.unparse(value) if value is not None else None
                failures.append(f"{name} is defined as {rendered!r}")

        defaults = _environment_defaults(config_source)
        if defaults != EXPECTED_ENV_DEFAULTS:
            failures.append(f"environment defaults are {defaults!r}")
        loaded_names = [
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        ]
        expected_loads = {
            "CHUNKFLOW_LIBERO_DATASET": 1,
            "CHUNKFLOW_REAL_DATA_DIR": 6,
            "CHUNKFLOW_PI05_BASE_CHECKPOINT": 6,
            "CHUNKFLOW_PAPER_REPO_ID": 2,
            "CHUNKFLOW_PAPER_DATASET_ROOT": 2,
            "CHUNKFLOW_SUPERVISED_CHECKPOINT": 2,
            "CHUNKFLOW_SUPERVISED_ASSETS": 1,
            "CHUNKFLOW_REAL_JOINT_STAGE1_CHECKPOINT": 1,
            "CHUNKFLOW_REAL_JOINT_STAGE2_CHECKPOINT": 1,
        }
        for name, expected in expected_loads.items():
            count = loaded_names.count(name)
            if count != expected:
                failures.append(f"{name} has {count} config uses; expected {expected}")

        configs = _train_config_calls(config_source)
        libero_config = configs.get("pi05_libero")
        if libero_config is None:
            failures.append("pi05_libero config is missing")
        else:
            libero_data = _keyword(libero_config, "data")
            if not isinstance(libero_data, ast.Call):
                failures.append("pi05_libero data config is missing")
            else:
                repo_id = _keyword(libero_data, "repo_id")
                try:
                    repo_id_value = ast.literal_eval(repo_id) if repo_id is not None else None
                except (ValueError, TypeError):
                    repo_id_value = ast.unparse(repo_id) if repo_id is not None else None
                if repo_id_value != "physical-intelligence/libero":
                    rendered = ast.unparse(repo_id) if repo_id is not None else None
                    failures.append(f"pi05_libero repo_id is {rendered!r}")
                root = _keyword(libero_data, "lerobot_root")
                if not isinstance(root, ast.Name) or root.id != "CHUNKFLOW_LIBERO_DATASET":
                    rendered = ast.unparse(root) if root is not None else None
                    failures.append(f"pi05_libero lerobot_root is {rendered!r}")
            weight_loader = _keyword(libero_config, "weight_loader")
            checkpoint = weight_loader.args[0] if isinstance(weight_loader, ast.Call) and weight_loader.args else None
            if not isinstance(checkpoint, ast.Name) or checkpoint.id != "CHUNKFLOW_PI05_BASE_CHECKPOINT":
                rendered = ast.unparse(checkpoint) if checkpoint is not None else None
                failures.append(f"pi05_libero checkpoint is {rendered!r}")

        for config_name, expected in EXPECTED_REAL_CONFIGS.items():
            data_factory, repo_id, action_space, downsampled, checkpoint_name = expected
            config = configs.get(config_name)
            if config is None:
                failures.append(f"retained config is missing: {config_name}")
                continue
            data = _keyword(config, "data")
            if not isinstance(data, ast.Call):
                failures.append(f"{config_name}: data config is missing")
                continue
            if _call_name(data.func) != data_factory:
                failures.append(f"{config_name}: data factory is {_call_name(data.func)!r}")

            actual_repo_id = _keyword(data, "repo_id")
            try:
                actual_repo_id_value = ast.literal_eval(actual_repo_id) if actual_repo_id is not None else None
            except (ValueError, TypeError):
                actual_repo_id_value = ast.unparse(actual_repo_id) if actual_repo_id is not None else None
            if actual_repo_id_value != repo_id:
                failures.append(f"{config_name}: repo_id is {actual_repo_id_value!r}")

            data_dir = _keyword(data, "rlds_data_dir")
            if not isinstance(data_dir, ast.Name) or data_dir.id != "CHUNKFLOW_REAL_DATA_DIR":
                rendered = ast.unparse(data_dir) if data_dir is not None else None
                failures.append(f"{config_name}: rlds_data_dir is {rendered!r}")

            actual_action_space = _keyword(data, "action_space")
            rendered_action_space = ast.unparse(actual_action_space) if actual_action_space is not None else None
            if rendered_action_space != action_space:
                failures.append(f"{config_name}: action_space is {rendered_action_space!r}")

            actual_downsampled = _keyword(data, "downsampled_and_repeated")
            try:
                actual_downsampled_value = (
                    ast.literal_eval(actual_downsampled) if actual_downsampled is not None else None
                )
            except (ValueError, TypeError):
                actual_downsampled_value = None
            if actual_downsampled_value is not downsampled:
                failures.append(
                    f"{config_name}: downsampled_and_repeated is {actual_downsampled_value!r}"
                )

            weight_loader = _keyword(config, "weight_loader")
            checkpoint = weight_loader.args[0] if isinstance(weight_loader, ast.Call) and weight_loader.args else None
            if not isinstance(checkpoint, ast.Name) or checkpoint.id != checkpoint_name:
                rendered = ast.unparse(checkpoint) if checkpoint is not None else None
                failures.append(f"{config_name}: checkpoint is {rendered!r}")

        if failures:
            self.fail("\n" + "\n".join(failures))

    def test_task_3b_local_libero_dataset_uses_lerobot_root(self) -> None:
        failures: list[str] = []
        config_source = _read("src/openpi/training/config.py")
        config_tree = ast.parse(config_source)

        for class_name in ("DataConfig", "DataConfigFactory"):
            class_node = next(
                (
                    node
                    for node in config_tree.body
                    if isinstance(node, ast.ClassDef) and node.name == class_name
                ),
                None,
            )
            root_field = next(
                (
                    node
                    for node in class_node.body
                    if isinstance(node, ast.AnnAssign)
                    and isinstance(node.target, ast.Name)
                    and node.target.id == "lerobot_root"
                ),
                None,
            ) if class_node is not None else None
            if root_field is None:
                failures.append(f"{class_name}.lerobot_root is missing")

        factory = next(
            (
                node
                for node in config_tree.body
                if isinstance(node, ast.ClassDef) and node.name == "DataConfigFactory"
            ),
            None,
        )
        create_base_config = next(
            (
                node
                for node in factory.body
                if isinstance(node, ast.FunctionDef) and node.name == "create_base_config"
            ),
            None,
        ) if factory is not None else None
        root_wiring = [
            keyword.value
            for node in ast.walk(create_base_config) if create_base_config is not None
            for keyword in (node.keywords if isinstance(node, ast.Call) else [])
            if keyword.arg == "lerobot_root"
        ]
        if [ast.unparse(value) for value in root_wiring] != ["self.lerobot_root"]:
            failures.append(
                f"DataConfigFactory root wiring is {[ast.unparse(value) for value in root_wiring]!r}"
            )

        data_loader_source = _read("src/openpi/training/data_loader.py")
        data_loader_tree = ast.parse(data_loader_source)
        expected_constructors = {
            "lerobot_dataset.LeRobotDatasetMetadata",
            "lerobot_dataset.LeRobotDataset",
        }
        constructor_calls = {
            _call_name(node.func): node
            for node in ast.walk(data_loader_tree)
            if isinstance(node, ast.Call) and _call_name(node.func) in expected_constructors
        }
        for constructor in expected_constructors:
            call = constructor_calls.get(constructor)
            if call is None:
                failures.append(f"{constructor} call is missing")
                continue
            root = _keyword(call, "root")
            if root is None or ast.unparse(root) != "data_config.lerobot_root":
                rendered = ast.unparse(root) if root is not None else None
                failures.append(f"{constructor} root is {rendered!r}")

        if failures:
            self.fail("\n" + "\n".join(failures))

    def test_task_3b_conversion_checkpoint_examples_do_not_duplicate_params(self) -> None:
        source = _read("examples/convert_jax_model_to_pytorch.py")
        documentation = ast.get_docstring(ast.parse(source), clean=False) or ""
        examples = [
            line.strip()
            for line in documentation.splitlines()
            if "python examples/convert_jax_model_to_pytorch.py" in line
            and "--checkpoint_dir" in line
        ]
        self.assertTrue(examples, "no documented --checkpoint_dir examples found")

        for example in examples:
            with self.subTest(example=example):
                arguments = shlex.split(example)
                checkpoint_index = arguments.index("--checkpoint_dir") + 1
                checkpoint_dir = arguments[checkpoint_index].rstrip("/")
                resolved_params = f"{checkpoint_dir}/params"
                self.assertFalse(
                    resolved_params.endswith("/params/params"),
                    f"documented checkpoint resolves to {resolved_params}",
                )

    def test_task_3b_conversion_copies_assets_from_checkpoint_root(self) -> None:
        source = _read("examples/convert_jax_model_to_pytorch.py")
        try:
            copy_assets = _load_isolated_function(
                source,
                "copy_checkpoint_assets",
                {"pathlib": pathlib, "shutil": shutil},
            )
        except AssertionError as error:
            self.fail(str(error))

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_parent = Path(temporary_directory)
            checkpoint = checkpoint_parent / "checkpoint"
            output = checkpoint_parent / "output"

            checkpoint_assets = checkpoint / "assets"
            checkpoint_assets.mkdir(parents=True)
            (checkpoint_assets / "sentinel.txt").write_text("checkpoint assets", encoding="utf-8")

            decoy_assets = checkpoint_parent / "assets"
            decoy_assets.mkdir()
            (decoy_assets / "decoy.txt").write_text("parent assets", encoding="utf-8")

            output_assets = output / "assets"
            output_assets.mkdir(parents=True)
            (output_assets / "stale.txt").write_text("stale assets", encoding="utf-8")

            copy_assets(str(checkpoint), str(output))

            copied_files = {
                path.relative_to(output_assets)
                for path in output_assets.rglob("*")
                if path.is_file()
            }
            self.assertEqual(copied_files, {Path("sentinel.txt")})
            self.assertEqual(
                (output_assets / "sentinel.txt").read_text(encoding="utf-8"),
                "checkpoint assets",
            )

    def test_task_3b_conversion_asset_copy_noops_without_source(self) -> None:
        source = _read("examples/convert_jax_model_to_pytorch.py")
        try:
            copy_assets = _load_isolated_function(
                source,
                "copy_checkpoint_assets",
                {"pathlib": pathlib, "shutil": shutil},
            )
        except AssertionError as error:
            self.fail(str(error))

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = root / "checkpoint-without-assets"
            checkpoint.mkdir()
            output_assets = root / "output" / "assets"
            output_assets.mkdir(parents=True)
            existing = output_assets / "existing.txt"
            existing.write_text("keep me", encoding="utf-8")

            copy_assets(str(checkpoint), str(output_assets.parent))

            self.assertEqual(existing.read_text(encoding="utf-8"), "keep me")

    def test_task_3b_conversion_calls_asset_helper_once(self) -> None:
        source = _read("examples/convert_jax_model_to_pytorch.py")
        converter = _top_level_function(source, "convert_pi0_checkpoint")
        self.assertIsNotNone(converter, "convert_pi0_checkpoint function not found")

        calls = [
            node
            for node in ast.walk(converter)
            if isinstance(node, ast.Call) and _call_name(node.func) == "copy_checkpoint_assets"
        ]
        self.assertEqual(len(calls), 1, f"copy_checkpoint_assets calls: {len(calls)}")
        self.assertEqual(
            [ast.unparse(argument) for argument in calls[0].args],
            ["checkpoint_dir", "output_path"],
        )

    def test_task_3b_success_map_cli_defaults_are_portable(self) -> None:
        source = _read("scripts/generate_libero_success_map.py")
        failures: list[str] = []

        try:
            repo_id = _argparse_default(source, "--repo-id")
        except (AssertionError, ValueError, SyntaxError) as error:
            failures.append(f"--repo-id: {error}")
        else:
            if repo_id != "physical-intelligence/libero":
                failures.append(f"--repo-id default is {repo_id!r}")

        try:
            root_default = _argparse_default_node(source, "--root")
        except (AssertionError, SyntaxError) as error:
            failures.append(f"--root: {error}")
        else:
            if not isinstance(root_default, ast.Call):
                failures.append(f"--root default is {ast.unparse(root_default)!r}")
            else:
                try:
                    arguments = [ast.literal_eval(argument) for argument in root_default.args]
                except (ValueError, TypeError):
                    arguments = []
                if (
                    _call_name(root_default.func) != "os.environ.get"
                    or arguments != ["CHUNKFLOW_LIBERO_DATASET", "datasets/libero_lerobot"]
                ):
                    failures.append(f"--root default is {ast.unparse(root_default)!r}")

        if failures:
            self.fail("\n" + "\n".join(failures))

    def test_task_3b_success_map_main_forwards_repo_id_and_root(self) -> None:
        constructor_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        class FakeLeRobotDataset:
            def __init__(self, *args, **kwargs):
                constructor_calls.append((args, kwargs))
                self._samples = [
                    {"episode_index": 3, "success": True},
                    {"episode_index": 4, "success": False},
                ]

            def __len__(self):
                return len(self._samples)

            def __getitem__(self, index):
                return self._samples[index]

        lerobot = _package_stub("lerobot")
        common = _package_stub("lerobot.common")
        datasets = _package_stub("lerobot.common.datasets")
        lerobot_dataset = types.ModuleType("lerobot.common.datasets.lerobot_dataset")
        lerobot_dataset.LeRobotDataset = FakeLeRobotDataset
        lerobot.common = common
        common.datasets = datasets
        datasets.lerobot_dataset = lerobot_dataset

        tqdm_module = types.ModuleType("tqdm")
        tqdm_module.tqdm = lambda iterable, **_: iterable
        numpy_module = types.ModuleType("numpy")

        module = _load_module_with_stubs(
            "scripts/generate_libero_success_map.py",
            "task_3b_generate_libero_success_map",
            {
                "lerobot": lerobot,
                "lerobot.common": common,
                "lerobot.common.datasets": datasets,
                "lerobot.common.datasets.lerobot_dataset": lerobot_dataset,
                "numpy": numpy_module,
                "tqdm": tqdm_module,
            },
        )

        repo_id = "example-org/libero-offline"
        root = "datasets/custom_libero_root"
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "success.json"
            result = module.main(repo_id, root, str(output_path))
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(result, 0)
        self.assertEqual(constructor_calls, [((), {"repo_id": repo_id, "root": root})])
        self.assertEqual(payload, {"3": True, "4": False})

    def test_task_3b_download_cli_defaults_to_cache(self) -> None:
        default = _argparse_default(_read("download_pi0.py"), "--output")
        self.assertIsNone(default, f"--output default is {default!r}")

    def test_task_3b_download_mode_selection_is_deterministic(self) -> None:
        source = _read("download_pi0.py")
        try:
            resolve_output = _load_isolated_function(source, "_resolve_model_output")
        except AssertionError as error:
            self.fail(str(error))

        cases = (
            (None, False, None),
            (None, True, "./checkpoints"),
            ("artifacts/models", False, "artifacts/models"),
            ("artifacts/models", True, "artifacts/models"),
        )
        for output, no_cache, expected in cases:
            with self.subTest(output=output, no_cache=no_cache):
                self.assertEqual(resolve_output(output, no_cache), expected)

    def test_task_3b_download_main_uses_resolved_model_output(self) -> None:
        source = _read("download_pi0.py")
        main = _top_level_function(source, "main")
        download_calls = [
            node
            for node in ast.walk(main) if main is not None
            if isinstance(node, ast.Call) and _call_name(node.func) == "download_pi0_model"
        ]
        if len(download_calls) != 1:
            self.fail(f"download_pi0_model calls: {len(download_calls)}")
        call = download_calls[0]
        output_arg = call.args[1] if len(call.args) > 1 else None
        if not isinstance(output_arg, ast.Call) or _call_name(output_arg.func) != "_resolve_model_output":
            rendered = ast.unparse(output_arg) if output_arg is not None else None
            self.fail(f"predefined-model output branch uses {rendered!r}")
        self.assertEqual(
            [ast.unparse(argument) for argument in output_arg.args],
            ["args.output", "args.no_cache"],
        )

    def test_task_3b_staged_joint_checkpoints_use_portable_prior_stages(self) -> None:
        source = _read("src/openpi/training/config.py")
        defaults = _environment_defaults(source)
        expected_defaults = {
            "CHUNKFLOW_REAL_JOINT_STAGE1_CHECKPOINT": "checkpoints/chunkflow_real_joint_stage1/params",
            "CHUNKFLOW_REAL_JOINT_STAGE2_CHECKPOINT": "checkpoints/chunkflow_real_joint_stage2/params",
        }
        self.assertEqual(
            {name: defaults.get(name) for name in expected_defaults},
            expected_defaults,
        )

        configs = _train_config_calls(source)
        expected_mapping = {
            "pi05_cytoderm11_joint_arm_move_chunkflow": "CHUNKFLOW_REAL_JOINT_STAGE1_CHECKPOINT",
            "pi05_cytoderm13_joint_arm_move_chunkflow": "CHUNKFLOW_REAL_JOINT_STAGE2_CHECKPOINT",
        }
        for config_name, checkpoint_name in expected_mapping.items():
            with self.subTest(config=config_name):
                config = configs[config_name]
                loader = _keyword(config, "weight_loader")
                checkpoint = loader.args[0] if isinstance(loader, ast.Call) and loader.args else None
                rendered = ast.unparse(checkpoint) if checkpoint is not None else None
                self.assertEqual(rendered, checkpoint_name)

    def test_task_3b_repository_relative_defaults_and_environment_mounts(self) -> None:
        failures: list[str] = []

        success_map_source = _read("scripts/generate_libero_success_map.py")
        expected_success_map_defaults = {"--output": "outputs/libero_success.json"}
        for option, expected in expected_success_map_defaults.items():
            try:
                default = _argparse_default(success_map_source, option)
            except (AssertionError, ValueError, SyntaxError) as error:
                failures.append(f"success map {option}: {error}")
                continue
            if default != expected or not isinstance(default, str) or Path(default).is_absolute():
                failures.append(f"success map {option} default is {default!r}")

        compose_source = _read("scripts/docker/compose.yml")
        mount_lines = [
            line.strip().removeprefix("- ")
            for line in compose_source.splitlines()
            if line.strip().startswith("- ${")
        ]
        for variable, target in EXPECTED_DOCKER_MOUNTS.items():
            candidates = [line for line in mount_lines if line.startswith(f"${{{variable}")]
            if len(candidates) != 1 or f":{target}" not in candidates[0]:
                failures.append(f"compose mount for {variable} is {candidates!r}")
        if "openpi_new_ymy" in compose_source:
            failures.append("compose retains a personal container name")

        download_source = _read("download_pi0.py")
        try:
            download_output = _argparse_default(download_source, "--output")
        except (AssertionError, ValueError, SyntaxError) as error:
            failures.append(f"download --output: {error}")
        else:
            if download_output is not None:
                failures.append(f"download --output default is {download_output!r}")
        if "pathlib.Path.home()" not in download_source:
            failures.append("download cache default does not use the current user's home")
        if "getpass" in download_source or "/" + "mnt/" in download_source:
            failures.append("download helper retains a host-specific cache fallback")

        conversion_source = _read("examples/convert_jax_model_to_pytorch.py")
        if "$HOME/.cache/openpi" not in conversion_source:
            failures.append("conversion examples do not use $HOME")
        if "/" + "home/$USER" in conversion_source or "/path/to/" in conversion_source:
            failures.append("conversion examples retain absolute placeholder paths")

        if failures:
            self.fail("\n" + "\n".join(failures))

    def test_task_3c_required_checkpoint_contracts(self) -> None:
        failures: list[str] = []

        evaluator_contracts = {
            "eval_code/real_robot_offline_eval.py": "DROIDDatasetEvaluator",
            "eval_code/real_robot_dataset_eval.py": "DROIDDatasetEvaluator",
            "eval_code/pi05_rtc_libero_eval.py": "Pi05RTCEvaluator",
        }
        for relative_path, evaluator_name in evaluator_contracts.items():
            path = ROOT / relative_path
            if not path.is_file():
                failures.append(f"checkpoint contract file is missing: {relative_path}")
                continue
            source = path.read_text(encoding="utf-8")
            try:
                checkpoint_default = _args_default(source, "checkpoint_dir")
            except (AssertionError, ValueError, SyntaxError) as error:
                failures.append(f"{relative_path}: {error}")
                continue
            if checkpoint_default != "":
                failures.append(f"{relative_path}: checkpoint default is {checkpoint_default!r}")
            if not _checkpoint_validation_precedes_evaluator(source, evaluator_name):
                failures.append(f"{relative_path}: checkpoint is not validated before evaluator initialization")

        for relative_path in NEW_EVALUATORS:
            path = ROOT / relative_path
            if not path.is_file():
                continue
            source = path.read_text(encoding="utf-8")
            for field_name, prefix in (("dataset_path", "datasets/real_robot"), ("output_dir", "outputs")):
                try:
                    default = _args_default(source, field_name)
                except (AssertionError, ValueError, SyntaxError) as error:
                    failures.append(f"{relative_path}: {error}")
                    continue
                if not isinstance(default, str) or not default.startswith(prefix):
                    failures.append(f"{relative_path}: {field_name} default is {default!r}")

        launcher = _read("eval_code/run_pi05_rtc_eval.sh")
        if 'CHECKPOINT_DIR="${CHUNKFLOW_CHECKPOINT:-}"' not in launcher:
            failures.append("RTC launcher does not default checkpoint from CHUNKFLOW_CHECKPOINT")
        if 'if [ -z "$CHECKPOINT_DIR" ]' not in launcher or "--checkpoint-dir is required" not in launcher:
            failures.append("RTC launcher does not reject an empty checkpoint")

        if failures:
            self.fail("\n" + "\n".join(failures))

    def test_task_3c_evaluator_defaults_match_public_configs(self) -> None:
        failures: list[str] = []
        for relative_path, expected_defaults in EXPECTED_EVALUATOR_DEFAULTS.items():
            path = ROOT / relative_path
            if not path.is_file():
                failures.append(f"missing evaluator file: {relative_path}")
                continue
            source = path.read_text(encoding="utf-8")
            for field_name, expected in expected_defaults.items():
                try:
                    actual = _args_default(source, field_name)
                except (AssertionError, ValueError, SyntaxError) as error:
                    failures.append(f"{relative_path}: {field_name}: {error}")
                    continue
                if actual != expected:
                    failures.append(
                        f"{relative_path}: {field_name} is {actual!r}; expected {expected!r}"
                    )

        if failures:
            self.fail("\n" + "\n".join(failures))

    def test_task_3c_checkpoint_validators_execute_before_heavy_imports(self) -> None:
        failures: list[str] = []
        for relative_path in EXPECTED_EVALUATOR_DEFAULTS:
            path = ROOT / relative_path
            if not path.is_file():
                failures.append(f"missing evaluator file: {relative_path}")
                continue
            source = path.read_text(encoding="utf-8")
            try:
                validator = _load_isolated_function(source, "_require_checkpoint_dir")
            except (AssertionError, NameError, SyntaxError) as error:
                failures.append(f"{relative_path}: {error}")
                continue

            for invalid in ("", "   "):
                try:
                    validator(invalid)
                except ValueError as error:
                    if str(error) != "--checkpoint-dir is required":
                        failures.append(f"{relative_path}: wrong error {error!s}")
                else:
                    failures.append(f"{relative_path}: accepted empty checkpoint {invalid!r}")

            expected = "checkpoints/example/run"
            try:
                actual = validator(expected)
            except Exception as error:  # pragma: no cover - reported as a test failure
                failures.append(f"{relative_path}: valid checkpoint raised {error!r}")
            else:
                if actual != expected:
                    failures.append(f"{relative_path}: validator returned {actual!r}")

        if failures:
            self.fail("\n" + "\n".join(failures))

    def test_task_3c_empty_checkpoint_stops_main_before_evaluator_side_effects(self) -> None:
        real_evaluators = (
            "eval_code/real_robot_offline_eval.py",
            "eval_code/real_robot_dataset_eval.py",
        )
        for relative_path in real_evaluators:
            with self.subTest(evaluator=relative_path):
                source = _read(relative_path)
                validator = _load_isolated_function(source, "_require_checkpoint_dir")
                events: list[str] = []
                args = types.SimpleNamespace(checkpoint_dir="")

                class FakeTyro:
                    @staticmethod
                    def cli(_args_type):
                        events.append("cli")
                        return args

                class FakeEvaluator:
                    def __init__(self, _args):
                        events.append("evaluator")

                    def run_evaluation(self):
                        events.append("run")

                main = _load_isolated_function(
                    source,
                    "main",
                    {
                        "Args": object,
                        "DROIDDatasetEvaluator": FakeEvaluator,
                        "_require_checkpoint_dir": validator,
                        "tyro": FakeTyro,
                    },
                )
                with self.assertRaisesRegex(ValueError, "^--checkpoint-dir is required$"):
                    main()
                self.assertEqual(events, ["cli"])

        rtc_source = _read("eval_code/pi05_rtc_libero_eval.py")
        rtc_validator = _load_isolated_function(rtc_source, "_require_checkpoint_dir")
        rtc_events: list[str] = []
        rtc_args = types.SimpleNamespace(checkpoint_dir="")

        class FakePath:
            @staticmethod
            def abspath(path):
                rtc_events.append("abspath")
                return path

            @staticmethod
            def join(*parts):
                rtc_events.append("join")
                return "/".join(parts)

        class FakeRTCEvaluator:
            def __init__(self, _args):
                rtc_events.append("evaluator")

        rtc_main = _load_isolated_function(
            rtc_source,
            "main",
            {
                "Args": object,
                "Pi05RTCEvaluator": FakeRTCEvaluator,
                "_require_checkpoint_dir": rtc_validator,
                "libero_path": "libero",
                "logging": types.SimpleNamespace(info=lambda *_: None, warning=lambda *_: None),
                "os": types.SimpleNamespace(path=FakePath),
                "set_libero_default_path": lambda _path: rtc_events.append("libero-config"),
            },
        )
        with self.assertRaisesRegex(ValueError, "^--checkpoint-dir is required$"):
            rtc_main(rtc_args)
        self.assertEqual(rtc_events, [])

    def test_task_3c_launcher_rejects_empty_checkpoint_before_side_effects(self) -> None:
        launcher = ROOT / "eval_code/run_pi05_rtc_eval.sh"
        environment = dict(os.environ)
        environment["CHUNKFLOW_CHECKPOINT"] = ""

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "must-not-be-created"
            result = subprocess.run(
                [
                    "bash",
                    str(launcher),
                    "--mode",
                    "baseline",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--checkpoint-dir is required", result.stdout + result.stderr)
            self.assertFalse(output_dir.exists())

    def test_task_3c_launcher_reports_missing_option_values(self) -> None:
        launcher = ROOT / "eval_code/run_pi05_rtc_eval.sh"
        value_options = (
            "--checkpoint-dir",
            "--task-suite",
            "--num-trials",
            "--output-dir",
            "--mode",
        )
        environment = dict(os.environ)
        environment["CHUNKFLOW_CHECKPOINT"] = "checkpoints/example"

        for option in value_options:
            for trailing_arguments in ((), ("--help",)):
                with self.subTest(option=option, trailing_arguments=trailing_arguments):
                    result = subprocess.run(
                        ["bash", str(launcher), option, *trailing_arguments],
                        cwd=ROOT,
                        env=environment,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=False,
                    )
                    output = result.stdout + result.stderr
                    self.assertEqual(result.returncode, 2)
                    self.assertIn(f"{option} requires a value", output)
                    self.assertNotIn("unbound variable", output)

    def test_task_3c_launcher_reports_malformed_json_with_current_name(self) -> None:
        launcher_source = _read("eval_code/run_pi05_rtc_eval.sh")
        marker = '        python -c "\n'
        start = launcher_source.index(marker) + len(marker)
        end = launcher_source.index('\n"', start)
        embedded_python = launcher_source[start:end]

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            report_dir = output_root / "broken-report"
            report_dir.mkdir()
            (report_dir / "evaluation_report.json").write_text("{broken", encoding="utf-8")
            embedded_python = embedded_python.replace("$OUTPUT_BASE", str(output_root))
            result = subprocess.run(
                [sys.executable, "-c", embedded_python],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        self.assertIn("broken-report", output)
        self.assertIn("Error:", output)
        self.assertNotIn("UnboundLocalError", output)

    def test_task_3c_evaluator_docs_use_public_names_and_cli_flags(self) -> None:
        failures: list[str] = []
        legacy_names = tuple(Path(path).name for path in OLD_EVALUATORS)
        underscore_flags = ("--checkpoint_dir", "--dataset_path", "--dataset_name", "--output_dir")

        for relative_path in NEW_EVALUATORS:
            path = ROOT / relative_path
            if not path.is_file():
                failures.append(f"missing evaluator file: {relative_path}")
                continue
            source = path.read_text(encoding="utf-8")
            documentation = ast.get_docstring(ast.parse(source), clean=False) or ""
            for legacy_name in legacy_names:
                if legacy_name in documentation:
                    failures.append(f"{relative_path}: documents legacy name {legacy_name}")
            for flag in underscore_flags:
                if flag in documentation:
                    failures.append(f"{relative_path}: documents underscore flag {flag}")

        if failures:
            self.fail("\n" + "\n".join(failures))

    def test_task_3b_segment_index_missing_path_is_exact_and_actionable(self) -> None:
        module = _load_truth_dataset_module()
        helper = getattr(module, "load_segment_index", None)
        self.assertIsNotNone(helper, "load_segment_index helper is missing")

        data_dir = "datasets/real_robot"
        repo_id = "chunkflow_real_cloth"
        expected_path = os.path.join(data_dir, f"{repo_id}_all_frame_idxs_add_path.pkl")
        exists_calls: list[str] = []
        loader_calls: list[str] = []

        def path_exists(path: str) -> bool:
            exists_calls.append(path)
            return False

        def loader(path: str):
            loader_calls.append(path)
            return object()

        with self.assertRaises(FileNotFoundError) as error:
            helper(data_dir, repo_id, path_exists=path_exists, loader=loader)

        self.assertEqual(exists_calls, [expected_path])
        self.assertEqual(loader_calls, [])
        self.assertIn(expected_path, str(error.exception))
        self.assertIn(
            "Generate it first or disable downsampled_and_repeated.",
            str(error.exception),
        )

    def test_task_3b_segment_index_existing_path_uses_injected_loader(self) -> None:
        module = _load_truth_dataset_module()
        helper = getattr(module, "load_segment_index", None)
        self.assertIsNotNone(helper, "load_segment_index helper is missing")

        repo_id = "chunkflow_real_cloth"
        sentinel = object()
        loader_calls: list[str] = []

        with tempfile.TemporaryDirectory() as temporary_directory:
            expected_path = os.path.join(
                temporary_directory,
                f"{repo_id}_all_frame_idxs_add_path.pkl",
            )
            Path(expected_path).touch()

            def loader(path: str):
                loader_calls.append(path)
                return sentinel

            result = helper(temporary_directory, repo_id, loader=loader)

        self.assertIs(result, sentinel)
        self.assertEqual(loader_calls, [expected_path])

    def test_task_3b_cartesian_dataset_calls_segment_index_helper(self) -> None:
        source = _read("src/openpi/training/truth_rlds_dataset.py")
        dataset_class = next(
            (
                node
                for node in ast.parse(source).body
                if isinstance(node, ast.ClassDef) and node.name == "TruthRldsDatasetCartesian"
            ),
            None,
        )
        initializer = next(
            (
                node
                for node in dataset_class.body
                if isinstance(node, ast.FunctionDef) and node.name == "__init__"
            ),
            None,
        ) if dataset_class is not None else None
        self.assertIsNotNone(initializer, "TruthRldsDatasetCartesian.__init__ is missing")

        calls = [
            node
            for node in ast.walk(initializer)
            if isinstance(node, ast.Call) and _call_name(node.func) == "load_segment_index"
        ]
        self.assertEqual(len(calls), 1, f"load_segment_index calls: {len(calls)}")
        self.assertEqual(
            [ast.unparse(argument) for argument in calls[0].args],
            ["data_dir", "repo_id"],
        )


if __name__ == "__main__":
    unittest.main()
