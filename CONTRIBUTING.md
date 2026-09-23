# Contributing to ChunkFlow

We welcome reproducible bug reports, documentation improvements, tests, and
focused changes to ChunkFlow. Contributions are accepted under the repository's
[Apache License 2.0](LICENSE).

ChunkFlow is derived from [Physical Intelligence's openpi](https://github.com/Physical-Intelligence/openpi).
Keep upstream attribution intact and do not remove copyright or license headers
from inherited or embedded third-party files. Changes that concern openpi alone,
rather than this derivative, may be better proposed to the upstream project.

## Before opening an issue

Use [ChunkFlow Issues](https://github.com/yangzhaojason/ChunkFlow/issues) for
reproducible defects, feature requests, and usage questions.

For a bug report, include:

- operating system, Python version, accelerator, and relevant driver/runtime
  versions;
- the exact config name and command;
- a minimal reproducer and complete traceback;
- dataset schema and tensor shapes without private data or credentials; and
- the commit tested.

Do not attach private datasets, checkpoints, tokens, machine-specific paths, or
other confidential artifacts.

## Development setup

```bash
git clone https://github.com/yangzhaojason/ChunkFlow.git
cd ChunkFlow
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

Create a focused branch, write a failing regression test before implementation,
and keep unrelated formatting or generated artifacts out of the change.

## Validation

Run the smallest relevant test while iterating, followed by the affected suite.
For release-document changes, run:

```bash
uv run pytest scripts/release_docs_test.py scripts/audit_release_test.py --confcutdir=scripts -q
uv run pytest src/openpi/training/portability_test.py --confcutdir=src/openpi/training -q
uv run python scripts/audit_release.py --root .
uv run ruff check scripts/release_docs_test.py scripts/audit_release.py scripts/audit_release_test.py
git diff --check
```

Model or data-pipeline changes should also run their colocated tests. Tests that
need external datasets or accelerators must document those prerequisites and
must not silently substitute private resources.

The release audit checks Git-tracked and non-ignored files for local workflow
artifacts, copied assistant instructions, private paths, credentials, symlinks,
and oversized files. These checks also run on pushes and pull requests.

## Pull requests

A pull request should explain the problem, summarize the chosen behavior, list
the commands used for verification, and call out compatibility or data-contract
changes. Keep the public API and the retained `openpi` Python namespace stable
unless the change explicitly requires a migration.

If modifying files covered by [third-party notices](THIRD_PARTY_NOTICES.md),
preserve their file headers and identify the inherited source in the pull
request. By submitting a contribution, you agree that it is distributed under
the repository license.
