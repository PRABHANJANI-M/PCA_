import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import joblib

# --- 1. Load dataset ---
url_train = "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra"
url_test  = "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes"

train = pd.read_csv(url_train, header=None)
test  = pd.read_csv(url_test, header=None)

df = pd.concat([train, test], ignore_index=True)
print("Dataset shape:", df.shape)

# --- 2. Separate features and labels ---
X = df.iloc[:, :-1]  # 64 pixels
y = df.iloc[:, -1]   # labels

# --- 3. Scale features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 4. PCA ---
pca_full = PCA().fit(X_scaled)
cumsum = np.cumsum(pca_full.explained_variance_ratio_)
n_components_95 = np.where(cumsum >= 0.95)[0][0] + 1
print("Number of PCA components to retain 95% variance:", n_components_95)

pca = PCA(n_components=n_components_95)
X_pca = pca.fit_transform(X_scaled)

# --- 5. KMeans Clustering ---
k = 10
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_pca)

# --- 6. Evaluate ---
ari = adjusted_rand_score(y, clusters)
nmi = normalized_mutual_info_score(y, clusters)
print(f"Adjusted Rand Index (ARI): {ari:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi:.4f}")

# --- 7. Save models ---
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca.pkl")
joblib.dump(kmeans, "kmeans_model.pkl")
print("Scaler, PCA, and KMeans models saved successfully!")

# --- 8. Cluster sizes ---
for i in range(k):
    print(f"Cluster {i} size: {(clusters == i).sum()}")
