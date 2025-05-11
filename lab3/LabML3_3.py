import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score,
    recall_score, f1_score, precision_recall_curve,
    roc_curve, auc
)

df = pd.read_csv("Titanic.csv")
df_clean = df.dropna()

columns_to_drop = [col for col in df_clean.columns if
                   df_clean[col].dtype == 'object' and col not in ['Sex', 'Embarked']]
df_clean = df_clean.drop(columns=columns_to_drop)

df_clean['Sex'] = df_clean['Sex'].map({'female': 0, 'male': 1})
df_clean['Embarked'] = df_clean['Embarked'].map({'C': 1, 'Q': 2, 'S': 3})
if 'PassengerId' in df_clean.columns:
    df_clean = df_clean.drop(columns=['PassengerId'])

# Делим данные
X = df_clean.drop(columns=['Survived'])
y = df_clean['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


def evaluate_model(name, model, X_test, y_test, y_pred, y_proba):
    print(f"\nModel: {name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred):.2f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.2f}")

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix: {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.plot(recall, precision, label=name)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()


lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_proba_lr = lr.predict_proba(X_test)[:, 1]
evaluate_model("Logistic Regression", lr, X_test, y_test, y_pred_lr, y_proba_lr)

svc = SVC(probability=True)
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)
y_proba_svc = svc.predict_proba(X_test)[:, 1]
evaluate_model("Support Vector Machine", svc, X_test, y_test, y_pred_svc, y_proba_svc)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
y_proba_knn = knn.predict_proba(X_test)[:, 1]
evaluate_model("K-Nearest Neighbors", knn, X_test, y_test, y_pred_knn, y_proba_knn)
