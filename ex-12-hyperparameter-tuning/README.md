# Exercise 12 — Hyperparameter Tuning

## 1. Aim

To tune the hyperparameters of a Support Vector Machine classifier for heart disease prediction using GridSearchCV with cross-validation.

## 2. Algorithm

1\. Load the dataset and separate the features from the target Heart Disease.

2\. Build a pipeline that scales the numerical features, one-hot encodes the categorical features and applies an SVC model.

3\. Split the data into training and testing sets in an 80:20 ratio with stratification.

4\. Define the parameter grid for C (0.1, 1, 10, 100), kernel (linear, rbf, poly) and gamma (scale, auto).

5\. Run GridSearchCV with 5-fold cross-validation using accuracy as the scoring metric.

6\. Display the best parameters and the best cross-validation score.

7\. Evaluate the best model on the test set using accuracy, precision, recall, F1 score and the classification report.

## 3. Program

```python
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
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
print(df.head())

X = df.drop("Heart Disease", axis=1)
y = df["Heart Disease"]

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

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("svm", SVC())
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

param_grid = {
    "svm__C": [0.1, 1, 10, 100],

    "svm__kernel": [
        "linear",
        "rbf",
        "poly"
    ],

    "svm__gamma": [
        "scale",
        "auto"
    ]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

print("\nTraining GridSearchCV...")

grid_search.fit(X_train, y_train)

print("\n====================================")
print("BEST PARAMETERS")
print("====================================")

print(grid_search.best_params_)

print("\nBest Cross Validation Score:")
print(grid_search.best_score_)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

print("Accuracy :",
      accuracy_score(y_test, y_pred))

print("Precision:",
      precision_score(y_test, y_pred, zero_division=0))

print("Recall   :",
      recall_score(y_test, y_pred, zero_division=0))

print("F1 Score :",
      f1_score(y_test, y_pred, zero_division=0))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

print("\nResult:")
print("SVM hyperparameters were successfully tuned")
print("using GridSearchCV.")
```

## 4. Output

The program was executed on the Heart disease dataset (heart_disease_dataset.csv) and produced the following output:

```text
Dataset Shape: (1000, 16)
   Age  Gender  ...   Chest Pain Type  Heart Disease
0   75  Female  ...   Atypical Angina              1
1   48    Male  ...    Typical Angina              0
2   53    Male  ...   Atypical Angina              1
3   69  Female  ...  Non-anginal Pain              0
4   62  Female  ...      Asymptomatic              0

[5 rows x 16 columns]

Training GridSearchCV...

====================================
BEST PARAMETERS
====================================
{'svm__C': 100, 'svm__gamma': 'auto', 'svm__kernel': 'rbf'}

Best Cross Validation Score:
0.9225
Accuracy : 0.915
Precision: 0.9066666666666666
Recall   : 0.8717948717948718
F1 Score : 0.8888888888888888

Classification Report:
              precision    recall  f1-score   support

           0       0.92      0.94      0.93       122
           1       0.91      0.87      0.89        78

    accuracy                           0.92       200
   macro avg       0.91      0.91      0.91       200
weighted avg       0.91      0.92      0.91       200


Result:
SVM hyperparameters were successfully tuned
using GridSearchCV.
```

## 5. Result

Thus, the Hyperparameter Tuning practical was implemented successfully. GridSearchCV selected C = 100, kernel = rbf and gamma = auto with a best cross-validation accuracy of 0.9225. On the test set the tuned model achieved an accuracy of 0.915, a precision of 0.9067, a recall of 0.8718 and an F1 score of 0.8889.
