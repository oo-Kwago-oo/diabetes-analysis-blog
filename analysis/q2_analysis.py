import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
df = pd.read_csv("../files/early_diabetes_data_cleaned.csv")

target = df['class']
features = df.drop('class', axis = 1).copy()

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.2, random_state = 42)

dtc = DecisionTreeClassifier(random_state=42)

param_grid = {
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10, 20],
    'max_leaf_nodes': [5, 10, 20],
}

gcv = GridSearchCV(dtc, param_grid=param_grid, n_jobs=-1)

gcv.fit(X_train, y_train)

opt_dtc = gcv.best_estimator_
opt_dtc.fit(X_train, y_train)

opt_dtc_proba = opt_dtc.predict_proba(X_test)

plt.figure(figsize=(12,8))
plot_tree(opt_dtc, feature_names=X_test.columns, class_names=["No", "Yes"], filled=True)
plt.savefig("tree.pdf", bbox_inches="tight")
plt.show()
