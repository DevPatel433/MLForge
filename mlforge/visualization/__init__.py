"""Plotting utilities (matplotlib is imported lazily)."""
from .animations import animate_kmeans, animate_optimizer_paths, plot_optimizer_paths
from .plots import (
    plot_benchmark_bars,
    plot_clusters,
    plot_coefficients,
    plot_confusion_matrix,
    plot_decision_boundary,
    plot_dendrogram,
    plot_elbow,
    plot_explained_variance,
    plot_feature_importance,
    plot_learning_curve,
    plot_learning_rate_vs_loss,
    plot_model_selection_curve,
    plot_pca,
    plot_precision_recall_curve,
    plot_regression_line,
    plot_regularization_path,
    plot_residuals,
    plot_roc_curve,
    plot_silhouette,
    plot_svm,
    plot_training_curves,
)
from .trees import plot_tree

__all__ = [n for n in dir() if n.startswith(("plot_", "animate_"))]
