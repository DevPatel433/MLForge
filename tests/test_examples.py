"""Every example script must run end-to-end (headless) and print something sensible."""
import glob
import os
import subprocess
import sys

import pytest

EXAMPLES = [p for p in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "examples", "*.py")))
            if not p.endswith("make_gallery.py")]          # the gallery is exercised separately (slow)


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: os.path.basename(p))
def test_example_runs(path, tmp_path):
    if "neural_network_example" in path:
        pytest.importorskip("sklearn")
    env = dict(os.environ, MPLBACKEND="Agg")
    out = subprocess.run([sys.executable, path], cwd=tmp_path, capture_output=True, text=True, env=env, timeout=300)
    assert out.returncode == 0, out.stderr[-800:]
    assert out.stdout.strip(), "example printed nothing"


def test_gradient_checking_example_reports_correctly(tmp_path):
    path = os.path.join(os.path.dirname(__file__), "..", "examples", "gradient_checking_example.py")
    out = subprocess.run([sys.executable, path], cwd=tmp_path, capture_output=True, text=True).stdout
    assert out.count("True") == 3
