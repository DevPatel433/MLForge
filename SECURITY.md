# Security policy

MLForge is a numerical library with no network access, no code generation, and a single runtime dependency (NumPy).

* **Pickle warning:** `BaseEstimator.save` / `load` use Python's `pickle`. **Never load a model file from an untrusted source** - unpickling can execute arbitrary code.
* To report a vulnerability, please use GitHub's *private vulnerability reporting* on this repository rather than a public issue.
* Supported versions: the latest minor release.
