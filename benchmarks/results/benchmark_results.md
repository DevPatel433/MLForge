# MLForge vs scikit-learn benchmark

| Algorithm | Data | Fit (MLForge) | Fit (sklearn) | Slowdown | Predict (MLForge) | Predict (sklearn) | Peak mem (MLForge / sklearn) | Quality (MLForge / sklearn) | Difference from scikit-learn |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LinearRegression (closed form) | regression | 1.1 ms | 1.5 ms | 0.8x | 0.2 ms | 0.3 ms | 0.3 / 0.5 MB | R2 0.8887 / 0.8887 | 0.0e+00 (max abs. Δ prediction) |
| LinearRegression (gradient descent) | regression | 186.3 ms | 1.6 ms | 115.1x | 0.2 ms | 0.4 ms | 0.2 / 0.5 MB | R2 0.8887 / 0.8887 | 1.6e-03 (max abs. Δ prediction) |
| Ridge | regression | 0.8 ms | 1.2 ms | 0.7x | 0.2 ms | 0.3 ms | 0.3 / 0.3 MB | R2 0.8886 / 0.8886 | 3.6e-15 (max abs. Δ prediction) |
| Lasso (coordinate descent) | regression | 1.6 ms | 1.4 ms | 1.2x | 0.1 ms | 0.3 ms | 0.5 / 0.3 MB | R2 0.8824 / 0.8824 | 1.8e-15 (max abs. Δ prediction) |
| ElasticNet | regression | 1.7 ms | 1.0 ms | 1.6x | 0.2 ms | 0.3 ms | 0.5 / 0.3 MB | R2 0.8831 / 0.8831 | 4.4e-15 (max abs. Δ prediction) |
| KNeighborsRegressor | regression | 0.1 ms | 0.5 ms | 0.3x | 58.3 ms | 5.0 ms | 0.0 / 0.0 MB | R2 0.5419 / 0.5419 | 0.0e+00 (max abs. Δ prediction) |
| DecisionTreeRegressor | regression | 40.2 ms | 17.1 ms | 2.4x | 1.2 ms | 0.5 ms | 2.4 / 0.1 MB | R2 0.3878 / 0.3811 | 5.9e+00 (max abs. Δ prediction) |
| RandomForestRegressor | regression | 5.24 s | 747.0 ms | 7.0x | 103.1 ms | 10.9 ms | 9.5 / 0.3 MB | R2 0.7047 / 0.7134 | 1.6e+00 (max abs. Δ prediction) |
| GradientBoostingRegressor | regression | 828.2 ms | 466.7 ms | 1.8x | 7.5 ms | 1.5 ms | 3.3 / 0.3 MB | R2 0.7510 / 0.7509 | 8.1e-01 (max abs. Δ prediction) |
| HistGradientBoostingRegressor | regression | 904.0 ms | 200.8 ms | 4.5x | 23.4 ms | 4.4 ms | 3.5 / 2.3 MB | R2 0.8392 / 0.8384 | 9.6e-01 (max abs. Δ prediction) |
| MLPRegressor | regression | 368.9 ms | 155.9 ms | 2.4x | 0.4 ms | 0.5 ms | 0.9 / 0.3 MB | R2 0.9611 / 0.8746 | 1.0e+01 (max abs. Δ prediction) |
| LogisticRegression (Newton) | cancer | 4.0 ms | 5.3 ms | 0.8x | 0.3 ms | 0.4 ms | 0.3 / 0.1 MB | accuracy 0.9766 / 0.9766 | 3.6e-07 (max abs. Δ probability) |
| LogisticRegression (gradient descent) | cancer | 422.4 ms | 5.0 ms | 84.7x | 0.2 ms | 0.4 ms | 0.3 / 0.1 MB | accuracy 0.9766 / 0.9766 | 7.3e-07 (max abs. Δ probability) |
| SoftmaxRegression | wine | 970.0 ms | 8.1 ms | 119.5x | 0.2 ms | 0.4 ms | 0.3 / 0.1 MB | accuracy 0.9818 / 0.9818 | 1.6e-05 (max abs. Δ probability) |
| Perceptron | binary | 365.2 ms | 3.3 ms | 111.4x | 0.3 ms | 0.5 ms | 0.0 / 0.1 MB | accuracy 0.8050 / 0.8217 | 2.0e-01 (label disagreement rate) |
| KNeighborsClassifier | digits | 0.5 ms | 0.8 ms | 0.6x | 50.1 ms | 6.6 ms | 0.1 / 0.1 MB | accuracy 0.9870 / 0.9870 | 0.0e+00 (max abs. Δ probability) |
| GaussianNB | cancer | 0.7 ms | 1.3 ms | 0.5x | 0.4 ms | 0.5 ms | 0.2 / 0.2 MB | accuracy 0.9532 / 0.9532 | 6.2e-15 (max abs. Δ probability) |
| DecisionTreeClassifier | cancer | 10.8 ms | 5.5 ms | 2.0x | 0.4 ms | 0.4 ms | 1.6 / 0.1 MB | accuracy 0.9415 / 0.9298 | 2.3e-02 (label disagreement rate) |
| RandomForestClassifier | digits | 3.70 s | 259.9 ms | 14.2x | 88.5 ms | 7.1 ms | 12.0 / 0.6 MB | accuracy 0.9740 / 0.9759 | 1.9e-02 (label disagreement rate) |
| GradientBoostingClassifier | cancer | 368.7 ms | 303.1 ms | 1.2x | 4.9 ms | 0.8 ms | 4.1 / 0.1 MB | accuracy 0.9708 / 0.9649 | 4.9e-01 (max abs. Δ probability) |
| HistGradientBoostingClassifier | digits | 3.59 s | 2.31 s | 1.6x | 95.6 ms | 33.8 ms | 4.0 / 6.4 MB | accuracy 0.9666 / 0.9666 | 7.4e-03 (label disagreement rate) |
| SVC (RBF kernel, SMO) | cancer | 39.1 ms | 3.4 ms | 11.4x | 0.5 ms | 1.6 ms | 2.7 / 0.1 MB | accuracy 0.9883 / 0.9883 | 0.0e+00 (label disagreement rate) |
| LinearSVC | binary | 33.2 ms | 2.8 ms | 11.7x | 0.2 ms | 0.4 ms | 0.1 / 0.1 MB | accuracy 0.8733 / 0.8733 | 0.0e+00 (label disagreement rate) |
| MLPClassifier | digits | 311.3 ms | 152.5 ms | 2.0x | 0.7 ms | 1.3 ms | 1.6 / 0.7 MB | accuracy 0.9592 / 0.9462 | 6.3e-02 (label disagreement rate) |
| KMeans (k-means++) | clusters | 14.4 ms | 11.1 ms | 1.3x | 0.0 ms | 0.0 ms | 0.3 / 0.2 MB | ARI 0.9197 / 0.9197 | 0.0e+00 (1 - adjusted Rand index) |
| AgglomerativeClustering (ward) | clusters | 251.0 ms | 68.0 ms | 3.7x | 0.0 ms | 0.0 ms | 36.0 / 10.1 MB | ARI 0.9019 / 0.9019 | 0.0e+00 (1 - adjusted Rand index) |
| DBSCAN | moons | 40.1 ms | 13.6 ms | 3.0x | 0.0 ms | 0.0 ms | 36.0 / 0.4 MB | ARI 0.9987 / 0.9987 | 0.0e+00 (1 - adjusted Rand index) |
| GaussianMixture (EM) | clusters | 22.1 ms | 9.8 ms | 2.2x | 0.0 ms | 0.0 ms | 0.4 / 0.4 MB | ARI 0.9195 / 0.9196 | 6.7e-03 (1 - adjusted Rand index) |
| PCA (SVD) | pca | 2.6 ms | 2.1 ms | 1.2x | 0.5 ms | 0.6 ms | 1.0 / 0.0 MB | explained variance 0.4758 / 0.4758 | 2.8e-16 (max abs. Δ variance ratio) |
