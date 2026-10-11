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
