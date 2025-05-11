import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("housing.csv", header=None)
split_data = data[0].str.split(expand=True).astype(float)
split_data.columns = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
    'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
]
X = split_data.drop(columns=["MEDV"])
y = split_data["MEDV"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


def plot_learning_curve(estimator, title, X, y, scoring='r2'):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 10), random_state=0
    )
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, 'o-', label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', label="Cross-validation score")
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel(scoring)
    plt.grid()
    plt.legend()
    plt.show()

plot_learning_curve(LinearRegression(), "Learning Curve: Linear Regression", X, y)
plot_learning_curve(Ridge(alpha=1.0), "Learning Curve: Ridge Regression", X, y)

models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5)
}

weights = {}
metrics = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    weights[name] = model.coef_
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    metrics[name] = {"MSE": mse, "R2": r2}

weights_df = pd.DataFrame(weights, index=X.columns)
metrics_df = pd.DataFrame(metrics).T
print("Метрики моделей:")
print(metrics_df)
print("\nКоэффициенты моделей:")
print(weights_df)

# Вывод:
# Лучшую точность показала линейная регрессия (R² ≈ 0.67)
# Ridge дал устойчивые результаты с немного худшей точностью.
# Lasso и ElasticNet выявили значимость признаков (некоторые обнулили).

