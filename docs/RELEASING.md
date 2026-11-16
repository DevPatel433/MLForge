# Releasing MLForge: a checklist for the steps that need *your* accounts

Everything in this repository is prepared so that each step below is a few clicks or one command. Nothing here can be done
from inside a sandbox, because each needs a GitHub, PyPI, Zenodo or JOSS account.

## 0. Before anything else: decide the package name

`mlforge` is **already taken on PyPI** by an unrelated project (it also installs a top-level `mlforge` module, so two such
packages would shadow each other in one environment). Options:

| Option | Command | Result |
|---|---|---|
| Keep the import name, publish as `mlforge-numpy` (the default in `pyproject.toml`) | nothing | `pip install mlforge-numpy`, `import mlforge` - **collides** with the other package if both are installed |
| Rename the import package too (recommended before the first release) | `python scripts/rename_package.py <newname>` | e.g. `pip install <newname>`, `import <newname>`; the script rewrites imports, docs, CI and renames the directory |

Names that were free on PyPI when checked: `mlforge-numpy`, `mlforge-scratch`, `mlforge-from-scratch`, `forgeml`, `mlforge-lab`
(availability changes; re-check at <https://pypi.org/project/NAME/>).

## 1. Put it on GitHub

```bash
python scripts/set_repo_url.py <your-github-username>      # replaces every "your-username" placeholder
git init && git add . && git commit -m "MLForge 0.3.0"
git branch -M main && git remote add origin https://github.com/<you>/<repo>.git && git push -u origin main
```
Then check the **Actions** tab. The CI matrix (Python 3.9-3.12; macOS and Windows) has never been executed by the author's tooling,
so expect to fix a failure or two - the evidence that it works elsewhere starts when this goes green.

## 2. Documentation site (GitHub Pages)

Settings -> Pages -> *Source: GitHub Actions*. The `docs` workflow builds `mkdocs build --strict` and deploys it.

## 3. PyPI

1. Create the project on PyPI and add a **trusted publisher** (GitHub repo + workflow `release.yml`).
2. Try TestPyPI first. `twine check dist/*` already passes locally (`make build`).
3. Tag and push: `git tag v0.3.0 && git push --tags` - the release workflow builds and attaches the artefacts; uncomment its publish step.

## 4. A citable DOI (Zenodo)

1. Sign in at <https://zenodo.org> with GitHub, then enable the repository under *GitHub*.
2. Publish a GitHub *release* (the tag from step 3). Zenodo archives it and mints a DOI (`.zenodo.json` supplies the metadata).
3. Put the DOI in `CITATION.cff` and add a badge to the README.

## 5. Peer-reviewed credential (JOSS / pyOpenSci)

`paper.md` and `paper.bib` follow the Journal of Open Source Software format. **Fill in the author block** (names, ORCIDs, affiliations).
Check the *current* submission requirements at <https://joss.theoj.org/about> before submitting - JOSS reviews look at substantial scholarly effort,
a public development history (issues, pull requests, releases over time), tests, documentation and community guidelines. A repository that has just been
created typically has not yet accumulated that history, so consider releasing publicly first and submitting once it has some. Acceptance is not guaranteed.
pyOpenSci is an alternative peer review for scientific Python packages.

## 6. Keep the evidence fresh

```bash
python tools/mutation_test.py        # mutation scores (about 25 min)
python tools/build_dashboard.py      # docs/correctness.md + badges
python docs/generate_api.py          # API reference
python examples/mnist_cnn.py --model cnn --epochs 5     # re-train the headline result
```
