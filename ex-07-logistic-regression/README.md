# Exercise 07 — Logistic Regression

## 1. Aim

To implement Logistic Regression for binary heart disease classification and to evaluate the model using accuracy, precision, recall, F1 score, the confusion matrix and the ROC-AUC score.

## 2. Algorithm

1\. Load the heart disease dataset and separate the features from the target Heart Disease.

2\. Identify the numerical and categorical columns and fill any missing values using the most frequent value.

3\. Convert the categorical columns into numeric form using LabelEncoder.

4\. Split the data into training and testing sets in an 80:20 ratio with random_state = 42.

5\. Train a Logistic Regression model with max_iter = 1000.

6\. Predict the test set and calculate accuracy, precision, recall and F1 score.

7\. Display the confusion matrix and the classification report.

8\. Plot the ROC curve and calculate the ROC-AUC score.

## 3. Program

```python
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

df = pd.read_csv("heart_disease_dataset.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

X = df.drop("Heart Disease", axis=1)
y = df["Heart Disease"]

print("\nFeatures:")
print(X.columns)

print("\nTarget:")
print(y.name)

categorical_columns = X.select_dtypes(
    include=["object"]
).columns

numerical_columns = X.select_dtypes(
    exclude=["object"]
).columns

print("\nCategorical Columns:")
print(list(categorical_columns))

print("\nNumerical Columns:")
print(list(numerical_columns))

imputer = SimpleImputer(strategy="most_frequent")

X[categorical_columns] = imputer.fit_transform(
    X[categorical_columns]
)

print("\nMissing values after imputation:")
print(X.isnull().sum())

label_encoders = {}

for column in categorical_columns:

    encoder = LabelEncoder()

    X[column] = encoder.fit_transform(
        X[column]
    )

    label_encoders[column] = encoder

print("\nData after Label Encoding:")
print(X.head())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data :", X_test.shape)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# minmax_scaler = MinMaxScaler()

# X_train_scaled = minmax_scaler.fit_transform(X_train)

# X_test_scaled = minmax_scaler.transform(X_test)

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)
model.fit(
    X_train_scaled,
    y_train
)

print("\nLogistic Regression model trained successfully.")

y_pred = model.predict(X_test_scaled)
# Probability of class 1
y_probability = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

print("\n================================")
print("CLASSIFICATION METRICS")
print("================================")

print("Accuracy  :", round(accuracy, 4))
print("Precision :", round(precision, 4))
print("Recall    :", round(recall, 4))
print("F1 Score  :", round(f1, 4))

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)

# ============================================
# 15. ROC Curve
# ============================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

# ============================================
# 16. Calculate AUC
# ============================================

roc_auc = auc(
    fpr,
    tpr
)

print("\nROC-AUC Score:",
      round(roc_auc, 4))

# ============================================
# 17. Plot ROC Curve
# ============================================

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label="Logistic Regression (AUC = {:.3f})".format(
        roc_auc
    )
)

# Random classifier line
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve - Heart Disease Prediction"
)

plt.legend()

plt.grid()
```

## 4. Output

The program was executed on the Heart disease dataset (heart_disease_dataset.csv) and produced the following output:

```text
First 5 rows:
   Age  Gender  ...   Chest Pain Type  Heart Disease
0   75  Female  ...   Atypical Angina              1
1   48    Male  ...    Typical Angina              0
2   53    Male  ...   Atypical Angina              1
3   69  Female  ...  Non-anginal Pain              0
4   62  Female  ...      Asymptomatic              0

[5 rows x 16 columns]

Dataset Shape:
(1000, 16)

Features:
Index(['Age', 'Gender', 'Cholesterol', 'Blood Pressure', 'Heart Rate',
       'Smoking', 'Alcohol Intake', 'Exercise Hours', 'Family History',
       'Diabetes', 'Obesity', 'Stress Level', 'Blood Sugar',
       'Exercise Induced Angina', 'Chest Pain Type'],
      dtype='object')

Target:
Heart Disease

Categorical Columns:
['Gender', 'Smoking', 'Alcohol Intake', 'Family History', 'Diabetes', 'Obesity', 'Exercise Induced Angina', 'Chest Pain Type']

Numerical Columns:
['Age', 'Cholesterol', 'Blood Pressure', 'Heart Rate', 'Exercise Hours', 'Stress Level', 'Blood Sugar']

Missing values after imputation:
Age                        0
Gender                     0
Cholesterol                0
Blood Pressure             0
Heart Rate                 0
Smoking                    0
Alcohol Intake             0
Exercise Hours             0
Family History             0
Diabetes                   0
Obesity                    0
Stress Level               0
Blood Sugar                0
Exercise Induced Angina    0
Chest Pain Type            0
dtype: int64

Data after Label Encoding:
   Age  Gender  ...  Exercise Induced Angina  Chest Pain Type
0   75       0  ...                        1                1
1   48       1  ...                        1                3
2   53       1  ...                        1                1
3   69       0  ...                        1                2
4   62       0  ...                        1                0

[5 rows x 15 columns]

Training data: (800, 15)
Testing data : (200, 15)

Logistic Regression model trained successfully.

================================
CLASSIFICATION METRICS
================================
Accuracy  : 0.865
Precision : 0.8493
Recall    : 0.7949
F1 Score  : 0.8212

Confusion Matrix:
[[111  11]
 [ 16  62]]

Classification Report:
              precision    recall  f1-score   support

           0       0.87      0.91      0.89       122
           1       0.85      0.79      0.82        78

    accuracy                           0.86       200
   macro avg       0.86      0.85      0.86       200
weighted avg       0.86      0.86      0.86       200


ROC-AUC Score: 0.9505
```

![image1.png](images/image1.png)

## 5. Result

Thus, the Logistic Regression practical was implemented successfully. The model achieved an accuracy of 86.5%, a precision of 0.8493, a recall of 0.7949, an F1 score of 0.8212 and a ROC-AUC score of 0.9505 on the test set.
