"""Replace every ``your-username`` placeholder with your real GitHub username (and optionally repo name).

    python scripts/set_repo_url.py octocat                 # github.com/octocat/mlforge
    python scripts/set_repo_url.py octocat --repo my-ml    # github.com/octocat/my-ml
    python scripts/set_repo_url.py octocat --dry-run       # show what would change
"""
import argparse
import os
import sys

TEXT_SUFFIXES = {".md", ".toml", ".yml", ".yaml", ".cff", ".json", ".txt", ".py", ".cfg", ".ipynb"}
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "site", "node_modules", ".ruff_cache"}


def rewrite(root, username, repo="mlforge", dry_run=False):
    changed = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if os.path.splitext(name)[1] not in TEXT_SUFFIXES:
                continue
            path = os.path.join(dirpath, name)
            try:
                text = open(path, encoding="utf-8").read()
            except UnicodeDecodeError:
                continue
            new = text.replace("your-username/mlforge", f"{username}/{repo}").replace("your-username", username)
            if new != text:
                changed.append(os.path.relpath(path, root))
                if not dry_run:
                    open(path, "w", encoding="utf-8").write(new)
    return changed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("username")
    ap.add_argument("--repo", default="mlforge")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    args = ap.parse_args(argv)
    changed = rewrite(args.root, args.username, args.repo, args.dry_run)
    print(("would change" if args.dry_run else "updated") + f" {len(changed)} files:")
    for c in changed:
        print("  ", c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
