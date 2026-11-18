"""Rename the importable package (e.g. to avoid clashing with another PyPI project called ``mlforge``).

    python scripts/rename_package.py forgeml                  # import forgeml, distribution "forgeml"
    python scripts/rename_package.py forgeml --dist forgeml-numpy
    python scripts/rename_package.py forgeml --brand ForgeML  # also rename the display name "MLForge"
    python scripts/rename_package.py forgeml --dry-run

It renames the ``mlforge/`` directory and rewrites every lowercase ``mlforge`` token (not adjoining letters or digits) in text files
(imports, ``python -m mlforge``, CI, docs, notebooks, entry points). Run the tests afterwards.
"""
import argparse
import keyword
import os
import re
import shutil
import sys

OLD = "mlforge"
TEXT_SUFFIXES = {".py", ".md", ".toml", ".yml", ".yaml", ".cff", ".json", ".txt", ".cfg", ".ipynb"}
TEXT_NAMES = {"Makefile"}
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "site", "node_modules", ".ruff_cache", ".ipynb_checkpoints", "scripts"}
SKIP_FILES = {"RELEASING.md"}          # describes the rename itself; the scripts/ directory is skipped for the same reason


def valid_name(name):
    return bool(re.fullmatch(r"[a-z][a-z0-9_]*", name)) and not keyword.iskeyword(name) and name not in sys.stdlib_module_names


def rename(root, new, dist=None, brand=None, dry_run=False):
    if not valid_name(new):
        raise ValueError(f"{new!r} is not a valid package name (lowercase letters, digits, underscore; not a stdlib module)")
    changed = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if name in SKIP_FILES or (os.path.splitext(name)[1] not in TEXT_SUFFIXES and name not in TEXT_NAMES):
                continue
            path = os.path.join(dirpath, name)
            try:
                text = open(path, encoding="utf-8").read()
            except UnicodeDecodeError:
                continue
            # underscores count as separators so data keys such as ``mlforge_fit_s`` are renamed consistently
            new_text = re.sub(rf"(?<![A-Za-z0-9]){OLD}(?![A-Za-z0-9])", new, text)
            if brand:
                new_text = new_text.replace("MLForge", brand)
            if name == "pyproject.toml":                                   # the distribution name is independent
                new_text = re.sub(r'^name = ".*"$', f'name = "{dist or new}"', new_text, count=1, flags=re.M)
            if new_text != text:
                changed.append(os.path.relpath(path, root))
                if not dry_run:
                    open(path, "w", encoding="utf-8").write(new_text)
    if not dry_run and os.path.isdir(os.path.join(root, OLD)):
        shutil.move(os.path.join(root, OLD), os.path.join(root, new))
    return changed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("new_name")
    ap.add_argument("--dist", help="PyPI distribution name (default: the new package name)")
    ap.add_argument("--brand", help='display name replacing "MLForge" in prose (optional)')
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    args = ap.parse_args(argv)
    changed = rename(args.root, args.new_name, args.dist, args.brand, args.dry_run)
    print(("would rewrite" if args.dry_run else "rewrote") + f" {len(changed)} files; package directory -> {args.new_name}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
