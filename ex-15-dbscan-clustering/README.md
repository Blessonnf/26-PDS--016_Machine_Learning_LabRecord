# Exercise 15 — Dbscan Clustering

## 1. Aim

To perform DBSCAN clustering on the heart disease dataset, identify the clusters and noise points, and visualize the clustering result using PCA.

## 2. Algorithm

1\. Load the dataset and separate the target.

2\. Identify the numerical and categorical features.

3\. Scale the numerical features and one-hot encode the categorical features.

4\. Apply DBSCAN with eps = 1.5 and min_samples = 5.

5\. Assign the cluster labels, identify the noise points and count the clusters.

6\. Apply PCA to reduce the data to two dimensions and plot the clustering result.

## 3. Program

```python
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

df = pd.read_csv("heart_disease_dataset.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 records:")
print(df.head())

X = df.drop("Heart Disease", axis=1)

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["object"]
).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

X_processed = preprocessor.fit_transform(X)

X_processed = X_processed

dbscan = DBSCAN(
    eps=1.5,
    min_samples=5
)

labels = dbscan.fit_predict(X_processed)

df["Cluster"] = labels

print("\nCluster Labels:")
print(df["Cluster"].value_counts().sort_index())

noise_points = df[df["Cluster"] == -1]

print("\nNumber of Noise Points:")
print(len(noise_points))

print("\nNoise Points:")
print(noise_points.head(10))

number_of_clusters = len(
    set(labels)
) - (1 if -1 in labels else 0)

print("\nNumber of Clusters:")
print(number_of_clusters)

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_processed)

plt.figure(figsize=(9, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=labels,
    cmap="viridis",
    s=40
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("DBSCAN Clustering")

plt.colorbar(label="Cluster")

plt.show()
```

## 4. Output

The program was executed on the Heart disease dataset (heart_disease_dataset.csv) and produced the following output:

```text
Dataset Shape: (1000, 16)

First 5 records:
   Age  Gender  ...   Chest Pain Type  Heart Disease
0   75  Female  ...   Atypical Angina              1
1   48    Male  ...    Typical Angina              0
2   53    Male  ...   Atypical Angina              1
3   69  Female  ...  Non-anginal Pain              0
4   62  Female  ...      Asymptomatic              0

[5 rows x 16 columns]

Cluster Labels:
Cluster
-1    1000
Name: count, dtype: int64

Number of Noise Points:
1000

Noise Points:
   Age  Gender  Cholesterol  ...   Chest Pain Type  Heart Disease Cluster
0   75  Female          228  ...   Atypical Angina              1      -1
1   48    Male          204  ...    Typical Angina              0      -1
2   53    Male          234  ...   Atypical Angina              1      -1
3   69  Female          192  ...  Non-anginal Pain              0      -1
4   62  Female          172  ...      Asymptomatic              0      -1
5   77    Male          309  ...      Asymptomatic              1      -1
6   64  Female          211  ...    Typical Angina              1      -1
7   60  Female          208  ...      Asymptomatic              1      -1
8   37  Female          317  ...  Non-anginal Pain              0      -1
9   63    Male          204  ...      Asymptomatic              1      -1

[10 rows x 17 columns]

Number of Clusters:
0
```

![image1.png](images/image1.png)

## 5. Result

Thus, the DBSCAN Clustering practical was implemented successfully. With eps = 1.5 and min_samples = 5, no point had five neighbours within that distance in the 27-dimensional feature space, so all 1000 records were labelled as noise and no cluster was formed.
