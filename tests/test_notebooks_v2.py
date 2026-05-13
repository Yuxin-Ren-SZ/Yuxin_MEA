"""Regression: notebooks/v2/*.ipynb must not import from the old packages.

After Phase 1 moved everything into `yuxin_mea.*`, every notebook under
`notebooks/v2/` must use the new namespace. This test parses each
notebook via `nbformat` and walks every code cell, asserting no string
of the form ``from dataset_manager|pipeline_manager|config_manager|pipeline_tasks\b``
appears.

Cheap insurance against a future PR adding a v2 notebook (or editing an
existing one) and forgetting to update an import.
"""

from __future__ import annotations

import re
from pathlib import Path

import nbformat
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_V2 = REPO_ROOT / "notebooks" / "v2"

# Match `from X` and `import X` where X is one of the old package names.
# `\b` prevents matching a substring like `pipeline_tasks_other`.
_FORBIDDEN = re.compile(
    r"^\s*(?:from|import)\s+"
    r"(dataset_manager|pipeline_manager|config_manager|pipeline_tasks)"
    r"\b",
    re.MULTILINE,
)


def _v2_notebooks() -> list[Path]:
    if not NOTEBOOKS_V2.is_dir():
        return []
    return sorted(NOTEBOOKS_V2.glob("*.ipynb"))


@pytest.mark.parametrize(
    "notebook_path",
    _v2_notebooks(),
    ids=lambda p: p.name,
)
def test_v2_notebook_uses_new_namespace(notebook_path: Path):
    """No code cell may import from the pre-Phase-1 package names."""
    nb = nbformat.read(notebook_path, as_version=4)
    offending: list[tuple[int, str]] = []
    for index, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        source = cell.source if isinstance(cell.source, str) else "".join(cell.source)
        for match in _FORBIDDEN.finditer(source):
            offending.append((index, match.group(0).strip()))
    assert not offending, (
        f"{notebook_path.name} contains forbidden imports:\n"
        + "\n".join(f"  cell {idx}: {line}" for idx, line in offending)
    )


def test_v2_notebooks_directory_exists():
    """Smoke: the directory exists and has at least one notebook."""
    assert NOTEBOOKS_V2.is_dir(), f"{NOTEBOOKS_V2} not found"
    assert _v2_notebooks(), "notebooks/v2/ contains no .ipynb files"


def test_no_v2_notebook_has_stale_outputs():
    """v2 notebooks must ship clean — empty outputs AND null execution_count.

    The pre-commit hook strips both; a notebook with `outputs: []` but
    `execution_count: 5` indicates someone ran cells locally and only
    cleared outputs by hand, leaving the notebook in a mixed state. Catch
    both to force a clean kernel-restart-and-clear-outputs before commit.
    """
    issues: list[str] = []
    for path in _v2_notebooks():
        nb = nbformat.read(path, as_version=4)
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            if cell.get("outputs"):
                issues.append(f"{path.name} cell {i} has populated outputs")
            if cell.get("execution_count") is not None:
                issues.append(f"{path.name} cell {i} has non-null execution_count")
    assert not issues, "\n".join(issues)
