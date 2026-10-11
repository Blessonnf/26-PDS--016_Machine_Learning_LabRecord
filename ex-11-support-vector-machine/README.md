# Exercise 11 — Support Vector Machine

## 1. Aim

To perform Support Vector Machine (SVM) classification using the Linear, Polynomial, RBF and Sigmoid kernels and to compare their performance.

## 2. Algorithm

1\. Load the dataset and separate the features from the target Heart Disease.

2\. Identify the numerical and categorical features.

3\. Create the preprocessing that scales the numerical features and one-hot encodes the categorical features.

4\. Split the data into training and testing sets in an 80:20 ratio with stratification.

5\. Train an SVM model for each of the four kernels.

6\. Calculate accuracy, precision, recall and F1 score for every kernel.

7\. Compare the kernels in a table and a bar plot.

## 3. Program

```python
# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

df = pd.read_csv("heart_disease_dataset.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 records:")
print(df.head())

X = df.drop("Heart Disease", axis=1)
y = df["Heart Disease"]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

print("\nNumerical Features:")
print(list(numeric_features))

print("\nCategorical Features:")
print(list(categorical_features))

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

kernels = ["linear", "poly", "rbf", "sigmoid"]

results = []

for kernel in kernels:

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("svm", SVC(kernel=kernel, random_state=42))
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results.append([
        kernel,
        accuracy,
        precision,
        recall,
        f1
    ])

    print("\n======================================")
    print("SVM Kernel:", kernel)
    print("======================================")

    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        zero_division=0
    ))

results_df = pd.DataFrame(
    results,
    columns=[
        "Kernel",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)

print("\n\nKernel Comparison:")
print(results_df)

plt.figure(figsize=(8, 5))

plt.bar(
    results_df["Kernel"],
    results_df["Accuracy"]
)

plt.xlabel("SVM Kernel")
plt.ylabel("Accuracy")
plt.title("SVM Performance with Different Kernels")
plt.ylim(0, 1)

plt.show()

# ------------------------------------------------------------
# Result
# ------------------------------------------------------------

print("\nResult:")
print("SVM classification was successfully performed")
print("using Linear, Polynomial, RBF and Sigmoid kernels.")
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

Numerical Features:
['Age', 'Cholesterol', 'Blood Pressure', 'Heart Rate', 'Exercise Hours', 'Stress Level', 'Blood Sugar']

Categorical Features:
['Gender', 'Smoking', 'Alcohol Intake', 'Family History', 'Diabetes', 'Obesity', 'Exercise Induced Angina', 'Chest Pain Type']

======================================
SVM Kernel: linear
======================================
Accuracy : 0.875
Precision: 0.8732394366197183
Recall   : 0.7948717948717948
F1 Score : 0.8322147651006712

Classification Report:
              precision    recall  f1-score   support

           0       0.88      0.93      0.90       122
           1       0.87      0.79      0.83        78

    accuracy                           0.88       200
   macro avg       0.87      0.86      0.87       200
weighted avg       0.87      0.88      0.87       200


======================================
SVM Kernel: poly
======================================
Accuracy : 0.925
Precision: 0.9315068493150684
Recall   : 0.8717948717948718
F1 Score : 0.9006622516556292

Classification Report:
              precision    recall  f1-score   support

           0       0.92      0.96      0.94       122
           1       0.93      0.87      0.90        78

    accuracy                           0.93       200
   macro avg       0.93      0.92      0.92       200
weighted avg       0.93      0.93      0.92       200


======================================
SVM Kernel: rbf
======================================
Accuracy : 0.905
Precision: 0.8933333333333333
Recall   : 0.8589743589743589
F1 Score : 0.8758169934640523

Classification Report:
              precision    recall  f1-score   support

           0       0.91      0.93      0.92       122
           1       0.89      0.86      0.88        78

    accuracy                           0.91       200
   macro avg       0.90      0.90      0.90       200
weighted avg       0.90      0.91      0.90       200


======================================
SVM Kernel: sigmoid
======================================
Accuracy : 0.825
Precision: 0.8028169014084507
Recall   : 0.7307692307692307
F1 Score : 0.7651006711409396

Classification Report:
              precision    recall  f1-score   support

           0       0.84      0.89      0.86       122
           1       0.80      0.73      0.77        78

    accuracy                           0.82       200
   macro avg       0.82      0.81      0.81       200
weighted avg       0.82      0.82      0.82       200



Kernel Comparison:
    Kernel  Accuracy  Precision    Recall  F1 Score
0   linear     0.875   0.873239  0.794872  0.832215
1     poly     0.925   0.931507  0.871795  0.900662
2      rbf     0.905   0.893333  0.858974  0.875817
3  sigmoid     0.825   0.802817  0.730769  0.765101
```

![image1.png](images/image1.png)

```text

Result:
SVM classification was successfully performed
using Linear, Polynomial, RBF and Sigmoid kernels.
```

## 5. Result

Thus, the Support Vector Machine practical was implemented successfully. The polynomial kernel performed best with an accuracy of 0.925 and an F1 score of 0.9007, followed by the RBF kernel (0.905), the linear kernel (0.875) and the sigmoid kernel (0.825).
