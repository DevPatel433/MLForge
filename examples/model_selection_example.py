"""Leak-free model selection: Pipeline + cross-validated grid search + learning curve."""

from mlforge import Pipeline
from mlforge.datasets import make_moons
from mlforge.model_selection import GridSearchCV, learning_curve, train_test_split
from mlforge.preprocessing import StandardScaler
from mlforge.svm import SVC

X, y = make_moons(400, noise=0.3, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

pipe = Pipeline([("scale", StandardScaler()), ("svm", SVC(random_state=0))])
search = GridSearchCV(pipe, {"svm__C": [0.1, 1, 10], "svm__gamma": [0.1, 0.5, 2.0]}, cv=5).fit(Xtr, ytr)

print("best parameters :", search.best_params_)
print(f"cross-validated : {search.best_score_:.3f}")
print(f"held-out test   : {search.score(Xte, yte):.3f}")
print("\nall candidates (mean CV accuracy):")
for params, score in sorted(zip(search.cv_results_['params'], search.cv_results_['mean_test_score']), key=lambda t: -t[1]):
    print(f"  {score:.3f}  {params}")

sizes, train, val = learning_curve(search.best_estimator_, Xtr, ytr, cv=4)
print("\nlearning curve (train size -> validation accuracy):")
for s, v in zip(sizes, val.mean(1)):
    print(f"  {int(s):4d} -> {v:.3f}")
