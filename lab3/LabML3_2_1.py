import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("Titanic.csv")

df_clean = df.dropna()

columns_to_drop = [col for col in df_clean.columns if
                   df_clean[col].dtype == 'object' and col not in ['Sex', 'Embarked']]
df_clean = df_clean.drop(columns=columns_to_drop)

df_clean['Sex'] = df_clean['Sex'].map({'female': 0, 'male': 1})

embarked_map = {'C': 1, 'Q': 2, 'S': 3}
df_clean['Embarked'] = df_clean['Embarked'].map(embarked_map)

if 'PassengerId' in df_clean.columns:
    df_clean = df_clean.drop(columns=['PassengerId'])

X = df_clean.drop(columns=['Survived'])
y = df_clean['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with Embarked: {accuracy:.2f}")
X_no_embarked = X.drop(columns=['Embarked'])
X_train_no_emb, X_test_no_emb, y_train_no_emb, y_test_no_emb = train_test_split(X_no_embarked, y, test_size=0.3,
                                                                                random_state=0)
clf_no_emb = LogisticRegression(max_iter=1000)
clf_no_emb.fit(X_train_no_emb, y_train_no_emb)
y_pred_no_emb = clf_no_emb.predict(X_test_no_emb)
accuracy_no_emb = accuracy_score(y_test_no_emb, y_pred_no_emb)
print(f"Accuracy without Embarked: {accuracy_no_emb:.2f}")
