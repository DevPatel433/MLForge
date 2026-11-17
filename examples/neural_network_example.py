"""Handwritten digits with the from-scratch MLP: training curves and confusion matrix."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits  # only used to load the data

from mlforge import visualization as viz
from mlforge.metrics import classification_report
from mlforge.neural_network import MLPClassifier
from mlforge.preprocessing import StandardScaler, train_test_split

X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
scaler = StandardScaler().fit(Xtr)

mlp = MLPClassifier(hidden_sizes=(64,), n_epochs=60, learning_rate=0.003, l2=1e-4, dropout=0.1,
                    early_stopping=True, patience=10, random_state=0)
mlp.fit(scaler.transform(Xtr), ytr)
pred = mlp.predict(scaler.transform(Xte))

print(mlp.model_.summary())
print(f"\nstopped after {mlp.n_epochs_run_} epochs, test accuracy = {mlp.score(scaler.transform(Xte), yte):.4f}\n")
print(classification_report(yte, pred))

viz.plot_training_curves(mlp.history_, save="digits_training_curves.png")
viz.plot_confusion_matrix(yte, pred, normalize="true", save="digits_confusion.png")
plt.close("all")
