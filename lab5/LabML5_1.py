import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve, roc_curve, auc
)

df = pd.read_csv("diabetes.csv")
X = df.drop(columns=["Outcome"])
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

logreg = LogisticRegression(max_iter=1000)
tree_clf = DecisionTreeClassifier(random_state=0)
logreg.fit(X_train, y_train)
tree_clf.fit(X_train, y_train)

y_pred_log = logreg.predict(X_test)
y_pred_tree = tree_clf.predict(X_test)


def get_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred)
    }


metrics_log = get_metrics(y_test, y_pred_log)
metrics_tree = get_metrics(y_test, y_pred_tree)
print("Логистическая регрессия:", metrics_log)
print("Решающее дерево:", metrics_tree)

depths = range(1, 21)
f1_scores = []
for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=0)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    f1_scores.append(f1_score(y_test, preds))

plt.figure(figsize=(8, 5))
plt.plot(depths, f1_scores, marker='o')
plt.xlabel("Max Depth")
plt.ylabel("F1 Score")
plt.title("F1 Score vs Tree Depth")
plt.grid(True)
plt.show()

optimal_depth = depths[np.argmax(f1_scores)]
optimal_tree = DecisionTreeClassifier(max_depth=optimal_depth, random_state=0)
optimal_tree.fit(X_train, y_train)

plt.figure(figsize=(16, 8))
plot_tree(optimal_tree, feature_names=X.columns, class_names=["No", "Yes"], filled=True)
plt.title(f"Decision Tree (max_depth={optimal_depth})")
plt.show()

importances = pd.Series(optimal_tree.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x=importances.values, y=importances.index)
plt.title("Feature Importances")
plt.xlabel("Importance")
plt.show()

y_scores = optimal_tree.predict_proba(X_test)[:, 1]
precision, recall, _ = precision_recall_curve(y_test, y_scores)
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 5))
plt.plot(recall, precision)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.show()

# Оптимальная глубина дерева: 4
# Построены: дерево, важность признаков, PR и ROC кривые.
