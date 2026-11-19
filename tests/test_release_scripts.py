"""The release helper scripts (placeholder replacement and package renaming)."""
import importlib.util
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def mini_repo(tmp_path):
    (tmp_path / "mlforge").mkdir()
    (tmp_path / "mlforge" / "__init__.py").write_text('"""MLForge"""\n__version__ = "1"\nfrom . import core\n')
    (tmp_path / "mlforge" / "core.py").write_text("import mlforge\nVALUE = 42  # mlforge internals\n")
    (tmp_path / "mlforge" / "__main__.py").write_text("print('hi')\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.py").write_text("from mlforge.core import VALUE\nimport mlforge.core as c\n")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "mlforge-numpy"\n[project.scripts]\nmlforge = "mlforge.__main__:main"\n'
                                              '[project.urls]\nHomepage = "https://github.com/your-username/mlforge"\n')
    (tmp_path / "README.md").write_text("# MLForge\n\n```bash\npython -m mlforge info\npip install mlforge\n```\nSee github.com/your-username/mlforge\n")
    (tmp_path / "CITATION.cff").write_text("repository-code: https://github.com/your-username/mlforge\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "RELEASING.md").write_text("`mlforge` is taken on PyPI\n")
    (tmp_path / "bench.py").write_text('row["mlforge_fit_s"] = 1\nfor tag in ("mlforge", "sklearn"):\n    key = f"{tag}_fit_s"\n')
    (tmp_path / "logo.png").write_bytes(b"\x89PNG your-username mlforge")
    return tmp_path


def test_set_repo_url_replaces_placeholders_only_in_text_files(mini_repo):
    mod = load("set_repo_url")
    changed = mod.rewrite(str(mini_repo), "octocat", "my-ml")
    assert sorted(changed) == ["CITATION.cff", "README.md", "pyproject.toml"]
    assert "github.com/octocat/my-ml" in (mini_repo / "README.md").read_text()
    assert "your-username" not in (mini_repo / "pyproject.toml").read_text()
    assert b"your-username" in (mini_repo / "logo.png").read_bytes()                      # binaries untouched


def test_set_repo_url_dry_run_changes_nothing(mini_repo):
    mod = load("set_repo_url")
    before = (mini_repo / "README.md").read_text()
    assert mod.rewrite(str(mini_repo), "octocat", dry_run=True)
    assert (mini_repo / "README.md").read_text() == before


def test_rename_package_rewrites_imports_directory_entry_points_and_keeps_prose(mini_repo):
    mod = load("rename_package")
    mod.rename(str(mini_repo), "forgeml", dist="forgeml-numpy")
    assert (mini_repo / "forgeml" / "core.py").exists() and not (mini_repo / "mlforge").exists()
    assert "import forgeml" in (mini_repo / "forgeml" / "core.py").read_text()
    assert "from forgeml.core import VALUE" in (mini_repo / "tests" / "test_a.py").read_text()
    pyproject = (mini_repo / "pyproject.toml").read_text()
    assert 'name = "forgeml-numpy"' in pyproject and 'forgeml = "forgeml.__main__:main"' in pyproject
    readme = (mini_repo / "README.md").read_text()
    assert "python -m forgeml info" in readme and "# MLForge" in readme               # brand name in prose is kept
    assert "`mlforge` is taken" in (mini_repo / "docs" / "RELEASING.md").read_text()   # the rename doc is left alone
    assert (mini_repo / "logo.png").read_bytes().count(b"mlforge") == 1                # binaries untouched
    # data keys containing the old name must be renamed consistently with the tags that build them
    assert 'row["forgeml_fit_s"]' in (mini_repo / "bench.py").read_text()


def test_rename_leaves_no_stale_name_behind(mini_repo):
    mod = load("rename_package")
    mod.rename(str(mini_repo), "forgeml")
    stale = []
    for dirpath, _, files in os.walk(mini_repo):
        for f in files:
            if f.endswith((".py", ".md", ".toml", ".cff")) and f != "RELEASING.md":
                text = open(os.path.join(dirpath, f)).read()
                if "mlforge" in text:
                    stale.append(f)
    assert stale == []


def test_rename_with_brand_and_dry_run_and_validation(mini_repo):
    mod = load("rename_package")
    assert mod.rename(str(mini_repo), "forgeml", dry_run=True) and (mini_repo / "mlforge").exists()
    mod.rename(str(mini_repo), "forgeml", brand="ForgeML")
    assert "# ForgeML" in (mini_repo / "README.md").read_text()
    for bad in ("Forge", "1abc", "my-pkg", "os", "class", ""):
        with pytest.raises(ValueError):
            mod.rename(str(mini_repo), bad)


def test_cli_entry_points_run(mini_repo):
    out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "set_repo_url.py"), "octocat", "--dry-run", "--root", str(mini_repo)],
                         capture_output=True, text=True)
    assert out.returncode == 0 and "would change 3 files" in out.stdout
    out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "rename_package.py"), "forgeml", "--dry-run", "--root", str(mini_repo)],
                         capture_output=True, text=True)
    assert out.returncode == 0 and "would rewrite" in out.stdout
