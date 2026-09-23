from __future__ import annotations

from pathlib import Path
import re
import subprocess
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    path = ROOT / relative_path
    assert path.is_file(), f"missing release document: {relative_path}"
    return path.read_text(encoding="utf-8")


def _markdown_section(markdown: str, heading: str, next_heading: str) -> str:
    assert heading in markdown, f"missing README section: {heading}"
    section = markdown.split(heading, maxsplit=1)[1]
    assert next_heading in section, f"missing README section after {heading}: {next_heading}"
    return section.split(next_heading, maxsplit=1)[0]


def _fenced_block(section: str, language: str) -> str:
    match = re.search(rf"```{language}\n(.*?)\n```", section, flags=re.DOTALL)
    assert match is not None, f"missing {language} fenced block"
    return match.group(1)


def test_readme_identifies_chunkflow_paper_and_authors() -> None:
    readme = _read("README.md")

    assert readme.startswith("# ChunkFlow\n")
    assert "ChunkFlow: Towards Continuity-Consistent Chunked Policy Learning" in readme
    assert "IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2026" in readme
    assert "accepted" in readme.lower()
    for author in (
        "Zhao Yang",
        "Yinan Shi",
        "Mingyuan Yao",
        "Wenyao Xue",
        "Yawei Jueluo",
        "Longjun Liu",
    ):
        assert author in readme
    assert "https://cytoderm-ai.github.io/" in readme


def test_readme_separates_chunkflow_from_upstream_openpi_authorship() -> None:
    readme = _read("README.md")
    lowered = readme.lower()

    assert "https://github.com/Physical-Intelligence/openpi" in readme
    assert "physical intelligence" in lowered
    assert "derived" in lowered or "derivative" in lowered
    assert "openpi" in lowered
    assert "python namespace" in lowered
    assert "π0.5" in readme
    assert "chunkflow authors" in lowered
    assert "π0/π0.5 authors" in lowered or "π0 and π0.5 authors" in lowered


def test_readme_states_supported_method_contract_without_exact_density_claims() -> None:
    readme = _read("README.md")
    lowered = readme.lower()

    for phrase in (
        "rtc overlap execution",
        "executed-history conditioning",
        "first-order continuity",
        "second-order continuity",
        "stopped target-relative boundary surrogate",
        "bc → step-wise awac",
        "jax only",
        "training only",
    ):
        assert phrase in lowered
    assert "flow-matching surrogate" in lowered
    assert "not an exact log-probability" in lowered
    assert "not an exact kl" in lowered
    assert "exact entropy" in lowered
    assert "docs/chunkflow_awac.md" in readme


def test_installation_uses_chunkflow_clone_without_submodule_commands() -> None:
    readme = _read("README.md")

    assert "git clone https://github.com/yangzhaojason/ChunkFlow.git" in readme
    assert "git@github.com:Physical-Intelligence/openpi.git" not in readme
    assert "--recurse-submodules" not in readme
    assert "git submodule" not in readme
    assert "submodule" not in readme.lower()
    for requirement in ("NVIDIA", "Ubuntu 22.04", "uv"):
        assert requirement in readme


def test_readme_has_reproducible_chunkflow_training_commands() -> None:
    readme = _read("README.md")

    for value in (
        "CHUNKFLOW_PAPER_REPO_ID",
        "CHUNKFLOW_PAPER_DATASET_ROOT",
        "CHUNKFLOW_PI05_BASE_CHECKPOINT",
        "uv run scripts/compute_norm_stats.py --config-name pi05_chunkflow_paper_bc",
        "uv run scripts/train.py pi05_chunkflow_paper_bc --exp-name chunkflow_bc --overwrite",
        "CHUNKFLOW_SUPERVISED_CHECKPOINT=checkpoints/pi05_chunkflow_paper_bc/chunkflow_bc/29999/params",
        "CHUNKFLOW_SUPERVISED_ASSETS=checkpoints/pi05_chunkflow_paper_bc/chunkflow_bc/29999/assets",
        "uv run scripts/train.py pi05_chunkflow_paper_awac --exp-name chunkflow_awac --overwrite",
    ):
        assert value in readme


def test_readme_uses_step_root_for_serving_and_evaluation() -> None:
    readme = _read("README.md")

    assert "--policy.config=pi05_chunkflow_paper_awac" in readme
    assert "--policy.dir=checkpoints/pi05_chunkflow_paper_awac/chunkflow_awac/29999" in readme
    assert "--checkpoint-dir checkpoints/pi05_chunkflow_paper_awac/chunkflow_awac/29999" in readme
    assert "params/" in readme
    assert "assets/" in readme
    for match in re.finditer(r"--(?:policy\.dir|checkpoint-dir)(?:=|\s+)(\S+)", readme):
        assert not match.group(1).rstrip("/\\").endswith("params")


def test_readme_paper_rtc_benchmark_commands_override_script_defaults() -> None:
    readme = _read("README.md")
    calvin = _markdown_section(readme, "### CALVIN", "### LIBERO")
    libero = _markdown_section(readme, "### LIBERO", "### Action smoothness")
    commands = (
        (_fenced_block(calvin, "bash"), "bash eval_code/eval_calvin_chunkflow.sh"),
        (_fenced_block(libero, "bash"), "eval_code/pi05_rtc_libero_eval.py"),
    )

    for command, entry_point in commands:
        assert entry_point in command
        assert "--overlap-size 8" in command
        assert "--replan-interval 2" in command


def test_readme_distinguishes_raw_server_from_stateful_rtc_execution() -> None:
    readme = _read("README.md")
    section = _markdown_section(readme, "### Raw actor policy server", "### CALVIN")
    lowered = " ".join(section.lower().split())

    assert "raw actor" in lowered
    assert "does not perform overlap blending" in lowered
    assert "does not maintain executed history" in lowered
    assert "not a complete chunkflow rtc runtime" in lowered
    assert "calvin and libero wrappers below" in lowered
    assert "episode state" in lowered
    assert "control loop" in lowered
    assert "stateful service" in lowered
    assert "raw actor websocket client" in lowered
    assert "not equivalent" in lowered

    example = _fenced_block(section, "python")
    compile(example, "README RTCPolicy example", "exec")
    for value in (
        "from openpi.policies import policy_config",
        "from openpi.policies.rtc_policy import RTCConfig, RTCPolicy",
        "from openpi.training import config as _config",
        'config = _config.get_config("pi05_chunkflow_paper_awac")',
        "base = policy_config.create_trained_policy(config, checkpoint_dir)",
        "rtc_config = RTCConfig.from_model_config(",
        "config.model",
        "overlap_size=8",
        "replan_interval=2",
        'blending_method="linear"',
        "track_metrics=True",
        "rtc_policy = RTCPolicy(base, rtc_config)",
        "rtc_policy.reset()",
        "rtc_policy.infer(observation)",
    ):
        assert value in example


def test_readme_documents_real_evaluation_entry_points_and_external_assets() -> None:
    readme = _read("README.md")

    for value in (
        "uv run scripts/serve_policy.py policy:checkpoint",
        "CALVIN_ROOT",
        "CALVIN_DATASET",
        "CHUNKFLOW_CHECKPOINT",
        "CHUNKFLOW_CONFIG=pi05_chunkflow_paper_awac",
        "bash eval_code/eval_calvin_chunkflow.sh",
        "uv run python eval_code/pi05_rtc_libero_eval.py",
        "--config pi05_chunkflow_paper_awac",
        "uv run python eval_code/eval_action_smoothness.py",
    ):
        assert value in readme
    lowered = readme.lower()
    assert "not bundled" in lowered
    for external_asset in ("external datasets", "model weights", "calvin"):
        assert external_asset in lowered


def test_readme_preserves_openpi_capability_navigation_and_backend_boundary() -> None:
    readme = _read("README.md")
    lowered = readme.lower()

    for capability in ("π0", "π0-FAST", "π0.5", "ALOHA", "DROID", "LIBERO"):
        assert capability in readme
    assert "remote inference" in lowered
    assert "pytorch" in lowered
    assert re.search(r"chunkflow awac[^\n]*not supported[^\n]*pytorch", lowered)


def test_owned_markdown_relative_links_resolve() -> None:
    files = subprocess.check_output(["git", "ls-files", "-z", "*.md"], cwd=ROOT).decode().split("\0")
    for relative_path in filter(None, files):
        source = ROOT / relative_path
        text = _read(relative_path)
        for raw_target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text):
            target = raw_target.strip().strip("<>")
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target = unquote(target.split("#", maxsplit=1)[0])
            resolved = source.parent / target
            assert resolved.exists(), f"broken relative link in {relative_path}: {raw_target}"


def test_contributing_targets_chunkflow_repository_and_preserves_upstream_credit() -> None:
    contributing = _read("CONTRIBUTING.md")

    assert contributing.startswith("# Contributing to ChunkFlow\n")
    assert "https://github.com/yangzhaojason/ChunkFlow/issues" in contributing
    assert "Physical Intelligence" in contributing
    assert "https://github.com/Physical-Intelligence/openpi" in contributing
    assert "scripts/release_docs_test.py" in contributing


def test_citation_cff_has_software_and_conference_metadata_without_guesses() -> None:
    citation = _read("CITATION.cff")

    for field in (
        "type: software",
        'title: "ChunkFlow"',
        "repository-code: https://github.com/yangzhaojason/ChunkFlow",
        "url: https://cytoderm-ai.github.io/",
        "license: Apache-2.0",
        "preferred-citation:",
        "  type: conference-paper",
        '  title: "ChunkFlow: Towards Continuity-Consistent Chunked Policy Learning"',
        '  collection-title: "IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)"',
        "  year: 2026",
    ):
        assert field in citation

    for given, family in (
        ("Zhao", "Yang"),
        ("Yinan", "Shi"),
        ("Mingyuan", "Yao"),
        ("Wenyao", "Xue"),
        ("Yawei", "Jueluo"),
        ("Longjun", "Liu"),
    ):
        author = rf"family-names: {family}\n\s+given-names: {given}"
        assert len(re.findall(author, citation)) == 2

    assert not re.search(r"^\s*(?:doi|date-released|start|end):", citation, flags=re.MULTILINE)


def test_notices_preserve_upstream_and_embedded_third_party_attribution() -> None:
    notice = _read("NOTICE")
    third_party = _read("THIRD_PARTY_NOTICES.md")

    assert "Physical Intelligence" in notice
    assert "openpi" in notice
    assert "https://github.com/Physical-Intelligence/openpi" in notice
    assert "Apache-2.0" in notice

    for value in (
        "Big Vision Authors",
        "src/openpi/models/gemma.py",
        "src/openpi/models/gemma_fast.py",
        "src/openpi/models/siglip.py",
        "Google LLC",
        "src/openpi/models/vit.py",
        "Hugging Face",
        "Google",
        "src/openpi/models_pytorch/transformers_replace/",
        "src/openpi/models_pytorch/transformers_replace/models/gemma/configuration_gemma.py",
        "src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py",
        "src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py",
        "src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py",
        "Apache-2.0",
        "file headers",
    ):
        assert value in third_party


def test_paper_to_code_maps_all_requested_equations_and_surrogate_boundaries() -> None:
    mapping = _read("docs/paper_to_code.md")
    lowered = mapping.lower()

    for equation in (3, 5, 7, 8, 9, 10, 12, 13, 14, 15, 16):
        assert f"Eq. ({equation})" in mapping
    for relative_path in (
        "src/openpi/policies/rtc_policy.py",
        "src/openpi/policies/rtc_policy_test.py",
        "src/openpi/models/chunkflow_history.py",
        "src/openpi/models/chunkflow_history_test.py",
        "src/openpi/models/chunkflow_losses.py",
        "src/openpi/models/chunkflow_losses_test.py",
        "src/openpi/models/chunkflow_awac.py",
        "src/openpi/models/chunkflow_awac_test.py",
        "src/openpi/models/chunkflow_critic.py",
        "src/openpi/models/chunkflow_critic_test.py",
        "src/openpi/training/chunkflow_awac_train.py",
        "src/openpi/training/chunkflow_awac_train_test.py",
    ):
        assert relative_path in mapping
        assert (ROOT / relative_path).is_file()
    assert "equation 14" in lowered
    assert "flow-matching surrogate" in lowered
    assert "equation 16" in lowered
    assert "reference-consistency surrogate" in lowered
    assert "not an exact" in lowered
    assert "stopped target-relative residual extension" in lowered
    assert "target-v ema" in lowered
    assert "stability extension" in lowered


def test_gitignore_ignores_only_repository_root_review_tmp() -> None:
    lines = {
        line.strip()
        for line in _read(".gitignore").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert "/tmp/" in lines
    assert "tmp/" not in lines
