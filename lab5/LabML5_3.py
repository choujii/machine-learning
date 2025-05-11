import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import f1_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

df = pd.read_csv("diabetes.csv")
X = df.drop(columns=["Outcome"])
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

rf = RandomForestClassifier(random_state=0)
param_dist = {
    'n_estimators': [50, 100, 200, 300, 400],
    'max_depth': [None] + list(range(2, 11)),
    'max_features': ['auto', 'sqrt', 'log2'],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rand_search = RandomizedSearchCV(
    rf, param_dist, n_iter=20, scoring='f1', cv=5, random_state=0, n_jobs=-1
)
start = time()
rand_search.fit(X_train, y_train)
rand_time = time() - start
best_rand_params = rand_search.best_params_
best_rand_score = rand_search.best_score_

print("RandomizedSearchCV:")
print("  Best F1 score (CV): {:.4f}".format(best_rand_score))
print("  Best params:", best_rand_params)
print("  Time: {:.2f} sec".format(rand_time))

n_estimators_vals = [50, 100, 200, 300, 400]
max_depth_vals = [None] + list(range(2, 11))
max_features_vals = ['auto', 'sqrt', 'log2']
min_samples_split_vals = [2, 5, 10]
min_samples_leaf_vals = [1, 2, 4]

space = {
    'n_estimators': hp.choice('n_estimators', n_estimators_vals),
    'max_depth': hp.choice('max_depth', max_depth_vals),
    'max_features': hp.choice('max_features', max_features_vals),
    'min_samples_split': hp.choice('min_samples_split', min_samples_split_vals),
    'min_samples_leaf': hp.choice('min_samples_leaf', min_samples_leaf_vals)
}


def objective(params):
    clf = RandomForestClassifier(**params, random_state=0)
    score = cross_val_score(clf, X_train, y_train, cv=5, scoring='f1', n_jobs=-1).mean()
    return {'loss': -score, 'status': STATUS_OK}


trials = Trials()
start = time()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=20, trials=trials, rstate=np.random.RandomState(0))
tpe_time = time() - start

best_tpe_params = {
    'n_estimators': n_estimators_vals[best['n_estimators']],
    'max_depth': max_depth_vals[best['max_depth']],
    'max_features': max_features_vals[best['max_features']],
    'min_samples_split': min_samples_split_vals[best['min_samples_split']],
    'min_samples_leaf': min_samples_leaf_vals[best['min_samples_leaf']]
}
best_tpe_score = -min(trials.losses())

print("\nHyperopt TPE Search:")
print("  Best F1 score (CV): {:.4f}".format(best_tpe_score))
print("  Best params:", best_tpe_params)
print("  Time: {:.2f} sec".format(tpe_time))
