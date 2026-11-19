import json
import os
import subprocess
import sys

import pytest

from mlforge.__main__ import list_estimators, main


def test_info_lists_estimators(capsys):
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    for name in ("KMeans", "SVC", "MLPClassifier", "GradientBoostingRegressor", "GridSearchCV"):
        assert name in out
    assert sum(len(v) for v in list_estimators().values()) >= 35


def test_version_flag_and_help(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0 and "mlforge" in capsys.readouterr().out
    assert main([]) == 0 and "usage" in capsys.readouterr().out.lower()


def test_demo_command(capsys):
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "test accuracy" in out and "macro avg" in out


def test_module_entry_point():
    out = subprocess.run([sys.executable, "-m", "mlforge", "--version"], capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.startswith("mlforge")


def test_estimators_have_no_sklearn_dependency():
    code = ("import sys, mlforge; import mlforge.linear_model, mlforge.tree, mlforge.svm, mlforge.neural_network, "
            "mlforge.cluster, mlforge.decomposition, mlforge.bayes, mlforge.neighbors, mlforge.model_selection; "
            "print('sklearn' in sys.modules, 'scipy' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "False False", out.stdout + out.stderr


def test_benchmark_quick_subset(tmp_path, capsys):
    pytest.importorskip("sklearn")
    rc = main(["benchmark", "--quick", "--only", "Ridge", "KNeighborsRegressor", "PCA", "--out", str(tmp_path), "--no-plots"])
    assert rc == 0
    rows = json.load(open(tmp_path / "benchmark_results.json"))
    assert {r["algorithm"] for r in rows} == {"Ridge", "KNeighborsRegressor", "PCA (SVD)"}
    for r in rows:
        assert r["status"] == "ok" and r["max_abs_diff"] < 1e-8
        assert r["mlforge_fit_s"] > 0 and r["sklearn_memory_mb"] >= 0
    md = open(tmp_path / "benchmark_results.md").read()
    assert md.count("|") > 20 and "Ridge" in md
    assert (tmp_path / "benchmark_results.csv").exists()


def test_benchmark_plots_are_written(tmp_path):
    pytest.importorskip("sklearn")
    pytest.importorskip("matplotlib")
    main(["benchmark", "--quick", "--only", "Ridge", "Lasso", "--out", str(tmp_path)])
    for f in ("benchmark_fit_time.png", "benchmark_difference.png", "benchmark_quality.png"):
        assert os.path.getsize(tmp_path / f) > 1000
