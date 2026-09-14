# PROJECT 2  HEART DISEASE PREDICTION (CLASSIFICATION)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier  # Changed Regressor to Classifier
from sklearn.metrics import classification_report

hrt = pd.read_csv("/heart.csv")


# DATA_train_test_split

X = hrt.drop("target", axis=1)
y = hrt["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
# print("accuracy",accuracy)

# now lets check for random forest classifier
model_rf = RandomForestClassifier(
    random_state=42
)  # Changed Regressor to Classifier and added random_state for reproducibility
model_rf.fit(X_train, y_train)

y_pred_rf = model_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
# print("accuracy_rf",accuracy_rf)


report_rf = classification_report(y_test, y_pred_rf)
print(report_rf)

report = classification_report(y_test, y_pred)
print(report)


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))

sns.heatmap(hrt.corr(), annot=True, cmap="coolwarm")

plt.show()
# heatmap is widely used for data visualization

# conclusion  =  according to the two models logistic regression is far more better than random forest
# because the data here is clean and features are linearly dependent on each other
