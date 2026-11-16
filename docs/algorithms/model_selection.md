# Model Selection & Pipelines

## Splitting and cross-validation
* `train_test_split(..., stratify=y)` keeps class proportions; invalid sizes raise informative errors.
* `KFold`, `StratifiedKFold`, `ShuffleSplit`: iterators of `(train_idx, test_idx)`. Folds partition the data exactly (tested).
* `cross_val_score(estimator, X, y, cv=5, scoring="roc_auc")`, `cross_val_predict`. Integer `cv` is stratified automatically for classifiers.
* Scorers: accuracy, balanced_accuracy, f1, f1_macro, precision, recall, roc_auc, neg_log_loss, r2, neg_mean_squared_error, neg_root_mean_squared_error, neg_mean_absolute_error - or any callable `(estimator, X, y) -> float`.

## Hyper-parameter search
```python
from mlforge.model_selection import GridSearchCV, RandomizedSearchCV
gs = GridSearchCV(KNeighborsClassifier(), {"k": [1, 5, 15], "weights": ["uniform", "distance"]}, cv=5).fit(X, y)
gs.best_params_, gs.best_score_, gs.cv_results_["mean_test_score"], gs.predict(X_new)
```
Candidates are fitted on cloned estimators (`clone()` copies hyper-parameters only, never fitted state). `RandomizedSearchCV` accepts lists or callables `rng -> value` (e.g. log-uniform).

## Diagnostics
`learning_curve` (score vs training size - diagnoses bias vs variance) and `validation_curve` (score vs one hyper-parameter); plot with `plot_model_selection_curve`.

## Pipeline
```python
from mlforge import Pipeline
pipe = Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler()), ("clf", LogisticRegression(C=1.0))])
GridSearchCV(pipe, {"clf__C": [0.01, 1, 100]}, cv=5).fit(X, y)    # nested parameters with step__param
```
Fitting a pipeline fits each transformer on the *training* fold only, which prevents the classic preprocessing **data leakage** inside cross-validation.

## Estimator protocol
Every estimator inherits `BaseEstimator`, which provides `get_params` / `set_params` / `clone` / `save` / `load`, a readable `repr`, automatic input validation
(2-D numeric finite arrays, consistent lengths, helpful messages) and `NotFittedError` when predicting before fitting.
