import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix

"""
Q1: Which symptoms are most strongly associated with diabetes, and why might these relationships exist?
"""

df = pd.read_csv("../files/early_diabetes_data_cleaned.csv")

target = df['class']
features = df.drop('class', axis = 1).copy()

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.2, random_state = 42)

rfc = RandomForestClassifier(random_state = 42)

param_grid = {
    'max_depth': [None,3, 5, 7, 10],
    'n_estimators': [50, 100, 150, 200,500],
}

gcv = GridSearchCV(rfc, param_grid = param_grid)
gcv.fit(X_train, y_train)

opt_rfc = gcv.best_estimator_
opt_rfc.fit(X_train, y_train)

# Predictions
y_pred_proba = opt_rfc.predict_proba(X_test)
y_pred = opt_rfc.predict(X_test)

# Metric : Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Metric : AUC score
# roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
roc_auc = roc_auc_score(y_test, y_pred)

# Rates
# fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
fpr, tpr, _ = roc_curve(y_test, y_pred)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)


