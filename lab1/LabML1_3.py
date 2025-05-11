from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

diabetes = datasets.load_diabetes()
X = diabetes.data[:, 2].reshape(-1, 1)
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

sklearn_model = LinearRegression()
sklearn_model.fit(X_train, y_train)
y_pred_sklearn = sklearn_model.predict(X_test)
X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]

beta = np.linalg.inv(X_train_b.T.dot(X_train_b)).dot(X_train_b.T).dot(y_train)
intercept_custom = beta[0]
slope_custom = beta[1]

y_pred_custom = X_test * slope_custom + intercept_custom

plt.figure(figsize=(12, 6))
plt.scatter(X_test, y_test, color='blue', alpha=0.6, label='Реальные значения')
plt.plot(X_test, y_pred_sklearn, color='red', linewidth=2, label='Scikit-Learn')
plt.plot(X_test, y_pred_custom, '--', color='green', linewidth=2, label='Собственная модель')
plt.xlabel('BMI (стандартизированный)', fontsize=12)
plt.ylabel('Прогрессирование диабета', fontsize=12)
plt.title('Сравнение моделей линейной регрессии', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


mae_sk = mean_absolute_error(y_test, y_pred_sklearn)
r2_sk = r2_score(y_test, y_pred_sklearn)
mape_sk = mean_absolute_percentage_error(y_test, y_pred_sklearn)

mae_cust = mean_absolute_error(y_test, y_pred_custom)
r2_cust = r2_score(y_test, y_pred_custom)
mape_cust = mean_absolute_percentage_error(y_test, y_pred_custom)

print("\nКоэффициенты моделей:")
print(f"[Scikit-Learn] Наклон: {sklearn_model.coef_[0]:.4f}, Смещение: {sklearn_model.intercept_:.4f}")
print(f"[Собственная]  Наклон: {slope_custom:.4f}, Смещение: {intercept_custom:.4f}\n")

print("Оценка качества моделей:")
print("{:<15} | {:<8} | {:<8} | {:<10}".format("Модель", "MAE", "R²", "MAPE (%)"))
print("-" * 50)
print("{:<15} | {:<8.2f} | {:<8.2f} | {:<10.2f}".format(
    "Scikit-Learn", mae_sk, r2_sk, mape_sk))
print("{:<15} | {:<8.2f} | {:<8.2f} | {:<10.2f}".format(
    "Собственная", mae_cust, r2_cust, mape_cust))

results = pd.DataFrame({
    'Реальное значение': y_test[:5],
    'Scikit-Learn': np.round(y_pred_sklearn[:5], 2),
    'Собственная модель': np.round(y_pred_custom[:5].flatten(), 2)
})
print("\nПример предсказаний:")
print(results.to_string(index=False))
