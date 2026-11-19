"""The shipped notebooks must be valid, executed, and free of error outputs.

(Re-generate with ``python notebooks/build_notebooks.py --execute``.)
"""
import glob
import os

import pytest

nbformat = pytest.importorskip("nbformat")

NOTEBOOKS = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "notebooks", "*.ipynb")))


def test_there_are_six_notebooks():
    assert len(NOTEBOOKS) == 6


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: os.path.basename(p))
def test_notebook_is_valid_executed_and_error_free(path):
    nb = nbformat.read(path, 4)
    nbformat.validate(nb)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert code_cells, "notebook has no code"
    for c in code_cells:
        assert c.execution_count is not None, f"unexecuted cell: {c.source[:60]!r}"
        assert not any(o.output_type == "error" for o in c.outputs), f"error output in {c.source[:60]!r}"
    assert any(o.output_type in ("display_data", "execute_result") for c in code_cells for o in c.outputs), "no figures"
    text = "\n".join(c.source for c in code_cells)
    assert "import sklearn" not in text and "from sklearn.model_selection" not in text    # sklearn only loads datasets


def test_notebooks_use_sklearn_only_for_datasets():
    for path in NOTEBOOKS:
        nb = nbformat.read(path, 4)
        for c in nb.cells:
            if c.cell_type != "code":
                continue
            for line in c.source.splitlines():
                if "sklearn" in line and not line.lstrip().startswith("#"):
                    assert "datasets" in line, f"{os.path.basename(path)}: unexpected sklearn use: {line}"
