"""Build (and optionally execute) the example notebooks.

    python notebooks/build_notebooks.py            # write .ipynb files
    python notebooks/build_notebooks.py --execute  # also run them and store outputs

Notebook source lives here as plain Python so it is reviewable and diff-friendly.
scikit-learn appears ONLY to load the bundled toy datasets and to cross-check results.
"""
import os
import sys

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

SETUP = """%matplotlib inline
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25})
import mlforge
from mlforge import visualization as viz
print("MLForge", mlforge.__version__)"""

NOTEBOOKS = {}

# --------------------------------------------------------------------------- 1
NOTEBOOKS["01_house_price_prediction"] = [
    md("# 01 - Regression: predicting house prices\n\nA from-scratch regression workflow on a **synthetic housing market** (so the true relationships are known and we can check what each model recovers): "
       "linear regression (closed form *and* gradient descent), polynomial features, Ridge / Lasso / Elastic Net, gradient boosting, and honest model selection with cross-validation."),
    code(SETUP),
    md("## 1. Data\nPrice depends linearly on size, bedrooms and age, non-linearly on location (distance to centre), and has noise. Two columns are pure noise."),
    code("""rng = np.random.RandomState(42)
n = 600
size = rng.uniform(40, 250, n)            # m^2
bedrooms = rng.randint(1, 6, n).astype(float)
age = rng.uniform(0, 60, n)
dist = rng.uniform(0.5, 30, n)            # km to centre
noise1, noise2 = rng.randn(n), rng.randn(n)
price = (1800*size + 12000*bedrooms - 900*age + 150000*np.exp(-dist/6) + rng.randn(n)*15000)/1000   # k$
names = ["size", "bedrooms", "age", "dist", "noise1", "noise2"]
X = np.column_stack([size, bedrooms, age, dist, noise1, noise2])
print(X.shape, "price range (k$):", price.min().round(), "-", price.max().round())"""),
    code("""from mlforge.preprocessing import train_test_split, StandardScaler
Xtr, Xte, ytr, yte = train_test_split(X, price, test_size=0.25, random_state=0)
sc = StandardScaler().fit(Xtr)                       # fit on TRAIN only (no leakage)
Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)"""),
    md("## 2. Linear regression two ways\nThe closed-form solution and gradient descent minimise the same loss, so on standardised features they must agree. (Plain GD converges geometrically here; *Adam* moves each parameter by about `learning_rate` per step, so it would need thousands of steps just to reach an intercept of ~300 - a good reminder to match the optimizer to the scale of the problem.)"),
    code("""from mlforge.linear_model import LinearRegression
from mlforge.metrics import r2_score, root_mean_squared_error
exact = LinearRegression().fit(Ztr, ytr)
gd = LinearRegression(solver="gd", learning_rate=0.1, n_iters=300).fit(Ztr, ytr)
print("closed form  R2 =", round(exact.score(Zte, yte), 4), " RMSE =", round(root_mean_squared_error(yte, exact.predict(Zte)), 2))
print("grad descent R2 =", round(gd.score(Zte, yte), 4))
print("max |coef difference|:", np.abs(exact.coef_ - gd.coef_).max().round(5))
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
viz.plot_learning_curve(gd.cost_history_, ax=ax[0], log_y=True, ylabel="MSE")
ax[1].scatter(yte, exact.predict(Zte), s=14, alpha=.6); ax[1].plot([yte.min(), yte.max()], [yte.min(), yte.max()], "r--")
ax[1].set_xlabel("true price"); ax[1].set_ylabel("predicted"); ax[1].set_title("Predicted vs true")
plt.tight_layout()"""),
    code("""_ = viz.plot_residuals(yte, exact.predict(Zte))"""),
    md("## 3. Regularisation and feature selection\nLasso should zero out the two noise columns. The path plot shows coefficients shrinking as alpha grows."),
    code("""from mlforge.linear_model import Ridge, Lasso, ElasticNet
for name, m in [("OLS", exact), ("Ridge(10)", Ridge(alpha=10).fit(Ztr, ytr)), ("Lasso(1)", Lasso(alpha=1.0).fit(Ztr, ytr)),
                ("ElasticNet", ElasticNet(alpha=0.3, l1_ratio=.5).fit(Ztr, ytr))]:
    print(f"{name:<11} R2={m.score(Zte, yte):.4f}  coefs:", {n: round(float(c), 1) for n, c in zip(names, m.coef_)})"""),
    code("""fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
viz.plot_regularization_path(lambda a: Ridge(alpha=a), Ztr, ytr, np.logspace(-2, 4, 30), ax=ax[0], title="Ridge path")
viz.plot_regularization_path(lambda a: Lasso(alpha=a), Ztr, ytr, np.logspace(-2, 1.5, 30), ax=ax[1], title="Lasso path - noise columns hit zero first")
plt.tight_layout()"""),
    md("## 4. Non-linearity: polynomial features and boosting\nPrice depends on `exp(-dist/6)`, which a straight line cannot capture."),
    code("""from mlforge.linear_model import PolynomialRegression
from mlforge.tree import GradientBoostingRegressor, RandomForestRegressor
results = {}
for name, m in {"Linear": LinearRegression(), "Polynomial(3)": PolynomialRegression(degree=3),
                "RandomForest": RandomForestRegressor(60, max_depth=8, oob_score=True, random_state=0),
                "GradientBoosting": GradientBoostingRegressor(200, learning_rate=0.05, max_depth=3, subsample=.8, random_state=0)}.items():
    m.fit(Ztr, ytr); results[name] = m.score(Zte, yte)
for k, v in results.items(): print(f"{k:<18} test R2 = {v:.4f}")"""),
    md("## 5. Honest model selection\nGrid search with 5-fold cross-validation in a `Pipeline` (so scaling is re-fitted inside every fold). With 450 training rows and only 6 features, regularisation has little to fix, so cross-validation should - and does - pick one of the *smallest* alphas: a model-selection procedure that can say \"no regularisation needed\" is working correctly."),
    code("""from mlforge import Pipeline
from mlforge.model_selection import GridSearchCV
pipe = Pipeline([("scale", StandardScaler()), ("model", Ridge())])
gs = GridSearchCV(pipe, {"model__alpha": [0.01, 0.1, 1, 10, 100, 1000]}, cv=5, scoring="neg_root_mean_squared_error").fit(Xtr, ytr)
print("best:", gs.best_params_, " CV RMSE =", round(-gs.best_score_, 2), " test R2 =", round(r2_score(yte, gs.predict(Xte)), 4))"""),
    code("""from mlforge.model_selection import learning_curve
sizes, tr, va = learning_curve(Ridge(alpha=1), Ztr, ytr, cv=5, scoring="r2")
viz.plot_model_selection_curve(sizes, tr, va, title="Learning curve (R2)")"""),
    md("## Takeaways\n* Closed form and gradient descent reach the same solution; GD needs scaled features.\n* Lasso identifies irrelevant features; Ridge shrinks without zeroing.\n* A linear model plateaus on non-linear structure; trees / boosting capture it.\n* Put preprocessing inside the `Pipeline` so cross-validation is leak-free."),
]

# --------------------------------------------------------------------------- 2
NOTEBOOKS["02_iris_classification"] = [
    md("# 02 - Classification on Iris\n\nEight classifiers on the classic 3-class flower dataset, decision boundaries in 2-D, ROC / confusion matrices, a tree you can read, and k-nearest-neighbours behaviour as k changes."),
    code(SETUP),
    code("""from sklearn.datasets import load_iris          # only used to load the data
iris = load_iris()
X, y, feats, classes = iris.data, iris.target, iris.feature_names, [str(c) for c in iris.target_names]
from mlforge.preprocessing import train_test_split, StandardScaler
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
sc = StandardScaler().fit(Xtr); Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
print(X.shape, classes)"""),
    md("## Compare many classifiers\nAll fitted from scratch; cross-validated accuracy (5-fold, stratified). Note the held-out test set has only 45 flowers, so one mistake moves accuracy by 2.2 points - differences of a few points between the top models are **not significant**; the cross-validated column is the more trustworthy one."),
    code("""from mlforge.linear_model import SoftmaxRegression, Perceptron
from mlforge.neighbors import KNeighborsClassifier
from mlforge.bayes import GaussianNB
from mlforge.tree import DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier
from mlforge.svm import SVC, LinearSVC
from mlforge.neural_network import MLPClassifier
from mlforge.model_selection import cross_val_score
models = {
    "Softmax regression": SoftmaxRegression(n_iters=500), "Perceptron": Perceptron(random_state=0),
    "KNN (k=5)": KNeighborsClassifier(5), "Gaussian NB": GaussianNB(),
    "Decision tree": DecisionTreeClassifier(max_depth=3), "Random forest": RandomForestClassifier(50, random_state=0),
    "Gradient boosting": GradientBoostingClassifier(60, max_depth=2, random_state=0),
    "SVM (RBF)": SVC(random_state=0), "MLP": MLPClassifier((16,), n_epochs=150, random_state=0)}
rows = []
for name, m in models.items():
    cv = cross_val_score(m, Ztr, ytr, cv=5)
    rows.append((name, cv.mean(), cv.std(), m.clone().fit(Ztr, ytr).score(Zte, yte)))
print(f"{'model':<20}{'cv acc':>10}{'+-':>8}{'test acc':>10}")
for r in sorted(rows, key=lambda r: -r[1]): print(f"{r[0]:<20}{r[1]:>10.3f}{r[2]:>8.3f}{r[3]:>10.3f}")"""),
    md("## Decision boundaries on two features\nUsing petal length and width (the informative pair), every model draws a different kind of boundary."),
    code("""P = X[:, 2:4]
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, (name, m) in zip(axes.ravel(), {"Softmax regression": SoftmaxRegression(n_iters=800, learning_rate=0.3), "KNN (k=7)": KNeighborsClassifier(7),
        "Decision tree": DecisionTreeClassifier(max_depth=3), "Random forest": RandomForestClassifier(40, random_state=0),
        "SVM (RBF)": SVC(C=3, random_state=0), "Gaussian NB": GaussianNB()}.items()):
    m.fit(P, y); viz.plot_decision_boundary(m, P, y, ax=ax, title=f"{name}  (acc {m.score(P, y):.2f})")
    ax.set_xlabel("petal length (cm)"); ax.set_ylabel("petal width (cm)")
plt.tight_layout()"""),
    md("## A decision tree you can read"),
    code("""tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
print(tree.export_text(feats))
viz.plot_tree(tree, feats, classes);"""),
    md("## Evaluation: confusion matrix, per-class report, feature importance"),
    code("""from mlforge.metrics import classification_report
rf = RandomForestClassifier(100, oob_score=True, random_state=0).fit(Xtr, ytr)
pred = rf.predict(Xte)
print(classification_report(yte, pred)); print("out-of-bag accuracy:", round(rf.oob_score_, 3))
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
viz.plot_confusion_matrix(yte, pred, labels=[0, 1, 2], ax=ax[0]); ax[0].set_xticklabels(classes); ax[0].set_yticklabels(classes)
viz.plot_feature_importance(rf.feature_importances_, feats, ax=ax[1]); plt.tight_layout()"""),
    md("## k in k-NN: bias vs variance\nSmall k memorises (jagged boundary, zero training error); large k oversmooths."),
    code("""fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
for ax, k in zip(axes, [1, 5, 25, 100]):
    knn = KNeighborsClassifier(k).fit(P, y); viz.plot_decision_boundary(knn, P, y, ax=ax, title=f"k={k}  train acc {knn.score(P, y):.2f}"); ax.get_legend().remove()
plt.tight_layout()"""),
]

# --------------------------------------------------------------------------- 3
NOTEBOOKS["03_wine_classification"] = [
    md("# 03 - Wine classification: scaling, PCA and SVMs\n\n13 chemical measurements, 3 cultivars. The lesson: **feature scaling decides whether distance- and gradient-based models work**, and PCA reveals why the problem is easy."),
    code(SETUP),
    code("""from sklearn.datasets import load_wine
d = load_wine(); X, y = d.data, d.target
print(X.shape, "feature scales (std):", np.round(X.std(0)[:6], 2), "...")
from mlforge.preprocessing import train_test_split, StandardScaler
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, random_state=0, stratify=y)"""),
    md("## Scaling matters\nThe proline column is ~1000x larger than the others, so it dominates Euclidean distances and wrecks unscaled KNN / SVM."),
    code("""from mlforge import Pipeline
from mlforge.neighbors import KNeighborsClassifier
from mlforge.svm import SVC
from mlforge.linear_model import SoftmaxRegression
for name, est in {"KNN": KNeighborsClassifier(5), "SVM (RBF)": SVC(random_state=0), "Softmax reg.": SoftmaxRegression(n_iters=300)}.items():
    raw = est.clone().fit(Xtr, ytr).score(Xte, yte)
    scaled = Pipeline([("sc", StandardScaler()), ("m", est.clone())]).fit(Xtr, ytr).score(Xte, yte)
    print(f"{name:<14} raw features: {raw:.3f}   standardised: {scaled:.3f}")"""),
    md("## PCA: why it is easy\nTwo components capture most variance and separate the cultivars."),
    code("""from mlforge.decomposition import PCA
Z = StandardScaler().fit_transform(X)
pca = PCA(None).fit(Z)
print("variance explained by first 2 / 3 / 5 components:", pca.explained_variance_ratio_[:2].sum().round(3), pca.explained_variance_ratio_[:3].sum().round(3), pca.explained_variance_ratio_[:5].sum().round(3))
print("components for 95% variance:", PCA(0.95).fit(Z).n_components_)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
viz.plot_explained_variance(pca, ax=ax[0]); viz.plot_pca(Z, y, n_components=2, ax=ax[1]); plt.tight_layout()"""),
    code("""_ = viz.plot_pca(Z, y, n_components=3)"""),
    md("## Classifiers in PCA space and kernel choice"),
    code("""P2 = PCA(2).fit_transform(Z)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
for ax, (name, m) in zip(axes, {"linear": SVC(kernel="linear", random_state=0), "RBF": SVC(C=3, random_state=0), "poly(3)": SVC(kernel="poly", C=3, random_state=0)}.items()):
    m.fit(P2, y); viz.plot_decision_boundary(m, P2, y, ax=ax, title=f"SVM {name}: acc {m.score(P2, y):.2f}  ({len(m.support_)} SVs)")
plt.tight_layout()"""),
    md("## Tuning the SVM with cross-validated grid search"),
    code("""from mlforge.model_selection import GridSearchCV
pipe = Pipeline([("sc", StandardScaler()), ("svm", SVC(random_state=0))])
gs = GridSearchCV(pipe, {"svm__C": [0.1, 1, 10], "svm__gamma": [0.003, 0.01, 0.03, 0.1]}, cv=4).fit(Xtr, ytr)
print("best:", gs.best_params_, " cv acc:", round(gs.best_score_, 3), " test acc:", round(gs.score(Xte, yte), 3))
res = gs.cv_results_; print("worst candidate cv acc:", res["mean_test_score"].min().round(3), "(hyper-parameters matter)")"""),
    md("## Probabilistic baseline\nGaussian Naive Bayes is trivially fast and strong here; its probabilities are inspectable."),
    code("""from mlforge.bayes import GaussianNB
from mlforge.metrics import log_loss, classification_report
nb = GaussianNB().fit(Xtr, ytr)
print("NB test acc:", round(nb.score(Xte, yte), 3), " log-loss:", round(log_loss(yte, nb.predict_proba(Xte)), 3))
print(classification_report(yte, nb.predict(Xte)))"""),
]

# --------------------------------------------------------------------------- 4
NOTEBOOKS["04_breast_cancer_prediction"] = [
    md("# 04 - Breast cancer prediction: probabilities, ROC and thresholds\n\nBinary diagnosis (malignant vs benign) from 30 cell-nucleus measurements. In medicine **accuracy is not enough**: we care about calibrated probabilities, the ROC / precision-recall trade-off and choosing a decision threshold."),
    code(SETUP),
    code("""from sklearn.datasets import load_breast_cancer
d = load_breast_cancer(); X, y = d.data, d.target       # y=1 benign, 0 malignant
print(X.shape, "class balance:", np.bincount(y))
from mlforge.preprocessing import train_test_split, StandardScaler
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, random_state=0, stratify=y)
sc = StandardScaler().fit(Xtr); Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)"""),
    md("## Logistic regression: three solvers, one optimum\nGradient descent, Adam and Newton / IRLS minimise the same regularised loss, so they must reach the same weights."),
    code("""from mlforge.linear_model import LogisticRegression
fits = {"GD": LogisticRegression(solver="gd", C=1.0, learning_rate=0.5, n_iters=12000),
        "Adam": LogisticRegression(solver="gd", C=1.0, optimizer="adam", learning_rate=0.05, n_iters=1500),
        "Newton": LogisticRegression(solver="newton", C=1.0)}
for n, m in fits.items(): m.fit(Ztr, ytr)
ref = fits["Newton"].coef_
for n, m in fits.items(): print(f"{n:<7} iterations={len(m.cost_history_):>5}  final loss={m.cost_history_[-1]:.6f}  max|w - w_newton|={np.abs(m.coef_-ref).max():.1e}  test acc={m.score(Zte, yte):.4f}")
viz.plot_learning_curve({k: m.cost_history_ for k, m in fits.items() if k != "Newton"}, log_y=True, title="Convergence of first-order solvers");"""),
    md("## Several models, one ROC plot"),
    code("""from mlforge.tree import RandomForestClassifier, GradientBoostingClassifier
from mlforge.svm import SVC
from mlforge.neural_network import MLPClassifier
from mlforge.metrics import roc_auc_score, average_precision_score
models = {"Logistic": fits["Newton"], "Random forest": RandomForestClassifier(100, random_state=0).fit(Ztr, ytr),
          "Boosting": GradientBoostingClassifier(100, random_state=0).fit(Ztr, ytr),
          "SVM": SVC(random_state=0).fit(Ztr, ytr),
          "MLP": MLPClassifier((32,), n_epochs=80, l2=1e-3, random_state=0).fit(Ztr, ytr)}
scores = {k: (m.predict_proba(Zte)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(Zte)) for k, m in models.items()}
for k, s in scores.items(): print(f"{k:<14} AUC={roc_auc_score(yte, s):.4f}  AP={average_precision_score(yte, s):.4f}")
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
viz.plot_roc_curve(yte, scores, ax=ax[0]); viz.plot_precision_recall_curve(yte, scores, ax=ax[1]); plt.tight_layout()"""),
    md("## Choosing a threshold\nMissing a malignant case (false *negative*) is far worse than a false alarm. Move the threshold to trade precision for recall."),
    code("""from mlforge.metrics import precision_score, recall_score, f1_score, confusion_matrix
p_malignant = 1 - models["Logistic"].predict_proba(Zte)[:, 1]
print(f"{'threshold':>10}{'precision':>11}{'recall':>9}{'missed cancers':>16}")
for t in [0.5, 0.3, 0.1, 0.05]:
    pred_malig = (p_malignant >= t).astype(int); true_malig = (yte == 0).astype(int)
    print(f"{t:>10}{precision_score(true_malig, pred_malig):>11.3f}{recall_score(true_malig, pred_malig):>9.3f}{int(((pred_malig == 0) & (true_malig == 1)).sum()):>16}")
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
for a, t in zip(ax, [0.5, 0.1]):
    viz.plot_confusion_matrix((yte == 0).astype(int), (p_malignant >= t).astype(int), labels=[0, 1], ax=a, title=f"threshold {t}")
plt.tight_layout()"""),
    md("## What drives the prediction?"),
    code("""names = np.array(d.feature_names)
rf = models["Random forest"]
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
viz.plot_feature_importance(rf.feature_importances_, names, top_k=8, ax=ax[0], title="Random-forest importance")
order = np.argsort(np.abs(models["Logistic"].coef_))[::-1][:8][::-1]
viz.plot_coefficients(models["Logistic"].coef_[order], names[order], ax=ax[1], title="Logistic coefficients (standardised)"); plt.tight_layout()"""),
    md("## Is the model overfitting? Regularisation sweep with cross-validation"),
    code("""from mlforge.model_selection import validation_curve
Cs = np.logspace(-4, 2, 7)
tr, va = validation_curve(LogisticRegression(solver="newton"), Ztr, ytr, "C", Cs, cv=5)
viz.plot_model_selection_curve(Cs, tr, va, xlabel="C (inverse regularisation)", log_x=True, title="Validation curve - Logistic regression");"""),
]

# --------------------------------------------------------------------------- 5
NOTEBOOKS["05_customer_segmentation"] = [
    md("# 05 - Customer segmentation (unsupervised learning)\n\nNo labels: discover customer groups from behaviour. Covers K-Means (with K-Means++ and the elbow / silhouette methods), hierarchical clustering with a dendrogram, DBSCAN for odd shapes and outliers, and Gaussian mixtures for *soft* assignments."),
    code(SETUP),
    md("## Synthetic customers with four hidden personas"),
    code("""rng = np.random.RandomState(7)
personas = {"Bargain hunters": ([25, 14, 3], [6, 3, 1]), "Loyal regulars": ([60, 40, 12], [8, 5, 2]),
            "Big spenders":  ([150, 8, 5], [20, 2, 1.5]), "Window shoppers": ([10, 3, 25], [4, 1.5, 6])}
X, truth = [], []
for k, (mu, sd) in enumerate(personas.values()):
    X.append(rng.randn(150, 3) * sd + mu); truth += [k] * 150
X, truth = np.vstack(X), np.array(truth)
cols = ["avg basket ($)", "orders / year", "site visits / week"]
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, (i, j) in zip(ax, [(0, 1), (0, 2), (1, 2)]): a.scatter(X[:, i], X[:, j], s=10, alpha=.5); a.set_xlabel(cols[i]); a.set_ylabel(cols[j])
plt.tight_layout()"""),
    md("## Scale first, then choose k\nDistance-based clustering needs standardised features. The **elbow** (inertia) and the **silhouette** both help pick k."),
    code("""from mlforge.preprocessing import StandardScaler
from mlforge.cluster import KMeans, elbow_curve, silhouette_curve
Z = StandardScaler().fit_transform(X)
ks = list(range(1, 10))
viz.plot_elbow(ks, elbow_curve(Z, ks), silhouette_curve(Z, range(2, 10)), title="Elbow (inertia) and silhouette");"""),
    code("""km = KMeans(4, n_init=10, random_state=0).fit(Z)
from mlforge.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score
print("silhouette:", round(silhouette_score(Z, km.labels_), 3), " Davies-Bouldin:", round(davies_bouldin_score(Z, km.labels_), 3),
      " ARI vs hidden personas:", round(adjusted_rand_score(truth, km.labels_), 3))
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
viz.plot_clusters(Z[:, :2], km.labels_, km.cluster_centers_[:, :2], ax=ax[0], title="K-Means (first two standardised features)")
viz.plot_silhouette(Z, km.labels_, ax=ax[1]); plt.tight_layout()"""),
    md("## Profile the segments\nTurn cluster centres back into business units."),
    code("""sc = StandardScaler().fit(X)
centres = sc.inverse_transform(km.cluster_centers_)
sizes = np.bincount(km.labels_)
print(f"{'cluster':>8}{'size':>6}" + "".join(f"{c:>20}" for c in cols))
for k in np.argsort(centres[:, 0]): print(f"{k:>8}{sizes[k]:>6}" + "".join(f"{v:>20.1f}" for v in centres[k]))"""),
    md("## Watch K-Means converge"),
    code("""print("iterations to converge:", km.n_iter_, "| inertia:", round(km.inertia_, 1))
km1 = KMeans(4, init="random", n_init=1, random_state=3).fit(Z[:, :2])
print("single random start inertia:", round(km1.inertia_, 1), "-> best of 10 K-Means++ starts:", round(KMeans(4, n_init=10, random_state=0).fit(Z[:, :2]).inertia_, 1))
# viz.animate_kmeans(km1, Z[:, :2], 'kmeans.gif')   # writes an animated GIF of the centroid trajectory"""),
    md("## Hierarchical clustering and the dendrogram"),
    code("""from mlforge.cluster import linkage, fcluster, AgglomerativeClustering
idx = rng.choice(len(Z), 60, replace=False)
Zl = linkage(Z[idx], "ward")
viz.plot_dendrogram(Zl, title="Ward dendrogram (60 customers)")
for m in ["single", "complete", "average", "ward"]:
    lab = AgglomerativeClustering(4, m).fit(Z).labels_
    print(f"{m:<9} ARI vs personas = {adjusted_rand_score(truth, lab):.3f}")"""),
    md("## DBSCAN: arbitrary shapes and outliers\nK-Means assumes round clusters. On two interleaved moons it fails; DBSCAN succeeds and flags noise."),
    code("""from mlforge.datasets import make_moons
from mlforge.cluster import DBSCAN
M, mt = make_moons(400, 0.07, random_state=0); M = np.vstack([M, [[2.2, 1.6], [-1.5, 1.6], [2, -1.2]]])   # plus 3 outliers
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
viz.plot_clusters(M, KMeans(2, random_state=0).fit(M).labels_, ax=ax[0], title="K-Means cuts the moons in half")
db = DBSCAN(0.15, 5).fit(M); viz.plot_clusters(M, db.labels_, ax=ax[1], title=f"DBSCAN: {db.n_clusters_} clusters, {(db.labels_ == -1).sum()} noise points"); plt.tight_layout()"""),
    md("## Gaussian mixtures: soft assignments and model selection by BIC"),
    code("""from mlforge.cluster import GaussianMixture
bics = [GaussianMixture(k, random_state=0).fit(Z).bic(Z) for k in range(1, 8)]
print("BIC by k:", np.round(bics, 0), "-> best k =", 1 + int(np.argmin(bics)))
gmm = GaussianMixture(4, random_state=0).fit(Z)
resp = gmm.predict_proba(Z)
conf = resp.max(1)
print("assignment confidence (max responsibility) - 1st / 5th percentile / median:", np.round(np.quantile(conf, [0.01, 0.05, 0.5]), 4))
print("least certain customer:", int(conf.argmin()), "->", np.round(resp[conf.argmin()], 3), "(a soft assignment, unlike K-Means)")
print("EM log-likelihood increased monotonically:", bool(np.all(np.diff(gmm.log_likelihood_history_) > -1e-9)))
print("ARI vs personas:", round(adjusted_rand_score(truth, gmm.predict(Z)), 3))"""),
]

# --------------------------------------------------------------------------- 6
NOTEBOOKS["06_mnist_digit_recognition"] = [
    md("# 06 - Handwritten digit recognition with a neural network\n\nA full pipeline: data -> from-scratch MLP (backprop, Adam, dropout, L2, early stopping) -> evaluation -> error analysis. Uses scikit-learn's bundled **8x8 digits** (1,797 images, instantly available offline); the last section shows the one-line change for the real 28x28 MNIST."),
    code(SETUP),
    code("""from sklearn.datasets import load_digits
d = load_digits(); X, y = d.data, d.target
print(X.shape, "pixels in [0,16]; classes:", np.unique(y))
fig, ax = plt.subplots(2, 10, figsize=(14, 3.2))
for i, a in enumerate(ax.ravel()): a.imshow(X[i].reshape(8, 8), cmap="gray_r"); a.set_title(y[i]); a.axis("off")
plt.tight_layout()"""),
    code("""from mlforge.preprocessing import train_test_split, StandardScaler
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=0, stratify=y)
sc = StandardScaler().fit(Xtr); Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)"""),
    md("## Train the network\n`MLPClassifier` is built on the library's own `Dense` / `ReLU` / `Dropout` layers, softmax cross-entropy and Adam."),
    code("""from mlforge.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_sizes=(128, 64), activation="relu", optimizer="adam", learning_rate=0.002,
                    l2=1e-4, dropout=0.1, n_epochs=80, batch_size=64, early_stopping=True, patience=10, random_state=0).fit(Ztr, ytr)
print("epochs run:", mlp.n_epochs_run_, "| train acc:", round(mlp.score(Ztr, ytr), 4), "| test acc:", round(mlp.score(Zte, yte), 4))
print(mlp.model_.summary())
viz.plot_training_curves(mlp.history_, title="Loss and accuracy per epoch");"""),
    md("## Where does it go wrong?"),
    code("""from mlforge.metrics import classification_report
pred = mlp.predict(Zte)
print(classification_report(yte, pred))
viz.plot_confusion_matrix(yte, pred, normalize="true", title="Normalised confusion matrix");"""),
    code("""wrong = np.where(pred != yte)[0][:10]
fig, ax = plt.subplots(1, max(len(wrong), 1), figsize=(1.6 * max(len(wrong), 1), 2.2))
for a, i in zip(np.atleast_1d(ax), wrong): a.imshow(Xte[i].reshape(8, 8), cmap="gray_r"); a.set_title(f"true {yte[i]}\\npred {pred[i]}", fontsize=8); a.axis("off")
plt.tight_layout()"""),
    md("## Does the optimizer matter? Same network, four optimizers\nPlain SGD needs far more epochs than Adam / RMSProp at the same budget."),
    code("""hist = {}
for opt, lr in [("sgd", 0.05), ("momentum", 0.02), ("rmsprop", 0.002), ("adam", 0.002)]:
    m = MLPClassifier((64,), optimizer=opt, learning_rate=lr, n_epochs=25, batch_size=64, random_state=0).fit(Ztr, ytr)
    hist[opt] = m.history_["loss"]; print(f"{opt:<9} final train loss {m.history_['loss'][-1]:.4f}  test acc {m.score(Zte, yte):.4f}")
viz.plot_learning_curve(hist, log_y=True, title="Training loss by optimizer", ylabel="cross-entropy");"""),
    md("## How does the MLP compare with simpler baselines?\nAlways compare against simple baselines before crediting model complexity. On small, clean image data a k-NN on a handful of principal components is a famously strong competitor."),
    code("""from mlforge.linear_model import SoftmaxRegression
from mlforge.decomposition import PCA
from mlforge.neighbors import KNeighborsClassifier
from mlforge import Pipeline
comparisons = {"Softmax regression (linear)": SoftmaxRegression(n_iters=300, learning_rate=0.1),
               "PCA(30) + KNN": Pipeline([("pca", PCA(30)), ("knn", KNeighborsClassifier(3))]),
               "MLP": mlp}
for k, m in comparisons.items():
    if m is not mlp: m.fit(Ztr, ytr)
    print(f"{k:<30} test acc = {m.score(Zte, yte):.4f}")"""),
    md("## Gradient check: proof that backprop is right\nBefore trusting any training curve, compare backprop with finite differences on a tiny network."),
    code("""from mlforge.neural_network import Sequential, Dense, Tanh, CrossEntropyLoss
from mlforge.utils import check_gradient
tiny = Sequential([Dense(8, 5, "xavier", 0), Tanh(), Dense(5, 10, "xavier", 1)]); ce = CrossEntropyLoss()
xb, yb = Ztr[:12, :8], ytr[:12]
f = lambda th: (tiny.set_flat_params(th), tiny.loss_and_grad(xb, yb, ce)[0])[1]
g = lambda th: (tiny.set_flat_params(th), tiny.loss_and_grad(xb, yb, ce)[1])[1]
ok, err = check_gradient(f, g, tiny.get_flat_params()); print("gradient check passed:", ok, "| relative error:", f"{err:.2e}")"""),
    md("## Real MNIST (28x28)\n```python\nfrom sklearn.datasets import fetch_openml      # needs internet, ~15 MB\nX, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)\nX, y = X / 255.0, y.astype(int)\n# then the same code: MLPClassifier((256, 128), ...).fit(X[:60000], y[:60000])\n```\nThe network, optimizer and tests are unchanged - only the data differ."),
]


def build(execute=False):
    import nbformat
    for name, cells in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook()
        nb.cells = cells
        nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}
        path = os.path.join(HERE, name + ".ipynb")
        if execute:
            from nbclient import NotebookClient
            NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": HERE}}).execute()
        nbformat.write(nb, path)
        print("wrote", name + ".ipynb", "(executed)" if execute else "")


if __name__ == "__main__":
    build(execute="--execute" in sys.argv)
