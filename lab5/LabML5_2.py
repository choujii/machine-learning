import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

df = pd.read_csv("diabetes.csv")
X = df.drop(columns=["Outcome"])
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


def get_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred)
    }


rf_base = RandomForestClassifier(random_state=0)
rf_base.fit(X_train, y_train)
y_pred_rf = rf_base.predict(X_test)
metrics_rf = get_metrics(y_test, y_pred_rf)
print("Random Forest (базовая модель):", metrics_rf)

depths = range(1, 21)
f1_depths = []
for d in depths:
    clf = RandomForestClassifier(max_depth=d, random_state=0)
    clf.fit(X_train, y_train)
    f1_depths.append(f1_score(y_test, clf.predict(X_test)))

plt.figure(figsize=(8, 5))
plt.plot(depths, f1_depths, marker='o')
plt.xlabel("Max Depth")
plt.ylabel("F1 Score")
plt.title("F1 Score vs Max Depth (Random Forest)")
plt.grid(True)
plt.show()

features_range = range(1, X.shape[1] + 1)
f1_features = []
for k in features_range:
    clf = RandomForestClassifier(max_features=k, random_state=0)
    clf.fit(X_train, y_train)
    f1_features.append(f1_score(y_test, clf.predict(X_test)))

plt.figure(figsize=(8, 5))
plt.plot(features_range, f1_features, marker='o')
plt.xlabel("Max Features")
plt.ylabel("F1 Score")
plt.title("F1 Score vs Max Features (Random Forest)")
plt.grid(True)
plt.show()
trees_range = range(10, 201, 10)
f1_trees = []
times = []
for n in trees_range:
    start = time()
    clf = RandomForestClassifier(n_estimators=n, random_state=0)
    clf.fit(X_train, y_train)
    duration = time() - start
    f1 = f1_score(y_test, clf.predict(X_test))
    f1_trees.append(f1)
    times.append(duration)

fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()
ax1.plot(trees_range, f1_trees, 'g-', marker='o', label='F1 Score')
ax2.plot(trees_range, times, 'b--', marker='x', label='Training Time')
ax1.set_xlabel("Number of Trees")
ax1.set_ylabel("F1 Score", color='g')
ax2.set_ylabel("Training Time (s)", color='b')
plt.title("F1 Score and Training Time vs n_estimators")
plt.grid(True)
plt.show()

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    reg_alpha=0.5,
    reg_lambda=1,
    objective='binary:logistic',
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=0
)

start = time()
xgb_model.fit(X_train, y_train)
xgb_duration = time() - start
y_pred_xgb = xgb_model.predict(X_test)
metrics_xgb = get_metrics(y_test, y_pred_xgb)

print("XGBoost:")
print("  Метрики:", metrics_xgb)
print("  Время обучения: {:.4f} сек.".format(xgb_duration))

# Вывод:
# Метод случайного леса показал стабильную производительность, глубина деревьев и кол-во признаков заметно влияют на F1.
# XGBoost дал высокое качество и быстрое обучение с подобранными параметрами.
