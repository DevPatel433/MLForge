"""Run the full MLForge vs scikit-learn benchmark suite.

    python benchmarks/run_benchmarks.py            # full run (a few minutes)
    python benchmarks/run_benchmarks.py --quick    # smoke run
    python benchmarks/run_benchmarks.py --only Ridge KMeans

Equivalent to ``python -m mlforge benchmark``. Results (markdown / csv / json
tables and PNG charts) are written to ``benchmarks/results/``.
"""
from mlforge.benchmarking import main

if __name__ == "__main__":
    main()
