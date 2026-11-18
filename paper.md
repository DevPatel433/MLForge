---
title: 'MLForge: machine learning from scratch, with measurable evidence of correctness'
tags:
  - Python
  - machine learning
  - NumPy
  - automatic differentiation
  - software testing
  - education
authors:
  - name: TODO - author name            # fill in before submitting (JOSS needs individual authors)
    orcid: 0000-0000-0000-0000          # TODO
    affiliation: 1
affiliations:
  - name: TODO - institution
    index: 1
date: 1 January 2027                    # TODO - submission date
bibliography: paper.bib
---

# Summary

MLForge is a Python library that implements 40 classical machine-learning estimators and a small deep-learning stack using only NumPy [@harris2020array]:
linear and logistic models with five solvers, decision trees, random forests, exact and histogram gradient boosting [@friedman2001greedy], k-nearest neighbours,
Naive Bayes, K-Means, hierarchical clustering, DBSCAN, Gaussian mixtures, PCA, kernel support-vector machines trained with sequential minimal optimisation [@platt1998smo],
and neural networks with dense, convolutional, pooling and LSTM layers. It also contains a reverse-mode automatic-differentiation engine [@baydin2018automatic],
pluggable optimizers (including Adam [@kingma2015adam]), model-selection utilities, 32 metrics and 25 plotting functions. The library itself has a single runtime dependency (NumPy).

The project's defining feature is not the list of algorithms but **how their correctness is established and reported**. Every gradient-based model is checked against
finite differences; deterministic algorithms are compared with scikit-learn [@pedregosa2011scikit] and SciPy [@virtanen2020scipy] at stated tolerances; mathematical laws are tested with
property-based testing [@maciver2019hypothesis]; and the test-suite itself is evaluated with mutation testing [@jia2011mutation]. A generated *correctness dashboard* reports each of these measurements,
together with benchmark results and a reproducible training run on the real MNIST dataset [@lecun1998gradient].

# Statement of need

Re-implementing machine-learning algorithms from first principles is a standard way to learn them, and many such implementations exist. Their evidence of correctness, however, is
usually that a model trains or that a plot looks reasonable - evidence that is weak, because a gradient that is wrong by a constant factor still decreases a loss, and a metric that
returns the wrong quantity still produces numbers in a plausible range. Researchers and educators who want to use or extend a from-scratch implementation need to know *which behaviours are verified and how*.

MLForge addresses this by making verification a first-class, quantitative output of the software. It targets (i) students and instructors who want readable reference implementations
with derivations (one documentation page per algorithm family), (ii) researchers who need a small, hackable code base in which a new method can be added alongside gradient-check and parity tests,
and (iii) software-testing practitioners interested in a worked example of mutation testing and property-based testing applied to numerical code.

# Design and verification

**Layers of evidence.** (1) `check_gradient` compares analytic and central-difference gradients; it is applied to every loss, to whole networks, to convolution (stride and padding variants),
pooling and back-propagation through time, with relative errors of order $10^{-10}$ against a $10^{-5}$ threshold. (2) A parity suite compares deterministic algorithms with scikit-learn and SciPy; where
randomness or tie-breaking makes bitwise agreement meaningless, tests assert the structural equivalent (leaf counts, neighbour *distances*, adjusted Rand index). (3) Property-based tests state laws such as
"folds partition the data" and "an unrestricted tree memorises any labelling of distinct points"; they found two real defects (a scaler dividing floating-point noise by itself on constant columns, and unbounded
recursion in tree growth when two feature values are one floating-point step apart), which are now regression-tested. (4) A dependency-free mutation-testing tool injects single faults into sampled code sites;
surviving mutants were triaged into genuine test gaps (fixed) and equivalent mutants. (5) A dashboard regenerates all of these measurements, and documentation numbers are produced by scripts rather than typed.

**Verified end to end.** A 5,258-parameter convolutional network trained from scratch on one CPU core for five epochs reaches 98.57% test accuracy on the 10,000-image MNIST test set
(about 97 s of training on one CPU core; a 109,386-parameter multilayer perceptron reaches 97.82%). The training script writes per-epoch metrics, the configuration and the weights, and an independent evaluation path reproduces the accuracy exactly.
MNIST files are fetched with SHA-256 verification.

**Performance, honestly.** Against scikit-learn on 29 algorithms, held-out quality is within 0.02 for all but one (a randomised neural-network case where the reference is under-trained at its default settings), while speed varies: closed-form and NumPy-friendly algorithms are on par,
tree ensembles are 2--14 times slower and Python-loop first-order solvers about two orders of magnitude slower. Histogram gradient boosting is 2.3 to 12.6 times faster than the library's exact boosting as the training set grows from 2,000 to 32,000 rows.
The library is intended for learning, verification and research prototyping rather than production throughput.

# Limitations

Dense arrays only; single-threaded; no sample or class weights, multi-output targets, or categorical splits. Randomised models match reference implementations statistically rather than exactly. Mutation scores are sampled estimates, not exhaustive.

# Acknowledgements

TODO - funding, contributors, and acknowledgement of the scikit-learn, NumPy and SciPy projects whose behaviour served as the reference.

# References
