# PROJECT 3 STUDENT PERFORMANCE PREDICTION (CLASSIFIER)


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier  # Changed Regressor to Classifier
from sklearn.metrics import classification_report

per = pd.read_csv("/content/StudentsPerformance.csv")

# Convert categorical features in X to numerical using one-hot encoding

per["avg_performer"] = (
    per["math score"] + per["reading score"] + per["writing score"]
) / 3

per["high_performer"] = (per["avg_performer"] >= 70).astype(int)

X = per[
    [
        "gender",
        "race/ethnicity",
        "lunch",
        "test preparation course",
    ]
]
y = per["high_performer"]
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2
)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("accuracy", accuracy)
