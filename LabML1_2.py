from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

diabetes = datasets.load_diabetes()
X = diabetes.data[:, 2].reshape(-1, 1)
y = diabetes.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression

model_sklearn = LinearRegression()
model_sklearn.fit(X_train, y_train)
y_pred_sklearn = model_sklearn.predict(X_test)

X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]

beta = np.linalg.inv(X_train_b.T.dot(X_train_b)).dot(X_train_b.T).dot(y_train)
slope_custom = beta[1]
intercept_custom = beta[0]

y_pred_custom = X_test * slope_custom + intercept_custom

print("[Scikit-Learn] Наклон: {:.4f}, Смещение: {:.4f}".format(model_sklearn.coef_[0], model_sklearn.intercept_))
print("[Собственная]  Наклон: {:.4f}, Смещение: {:.4f}\n".format(slope_custom, intercept_custom))

plt.figure(figsize=(12, 6))
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Реальные значения')
plt.plot(X_test, y_pred_sklearn, color='red', linewidth=2, label='Scikit-Learn')
plt.plot(X_test, y_pred_custom, '--', color='green', linewidth=2, label='Собственная модель')
plt.xlabel('BMI (стандартизированный)', fontsize=12)
plt.ylabel('Прогрессирование диабета', fontsize=12)
plt.title('Сравнение моделей линейной регрессии', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

results = pd.DataFrame({
    'Реальное значение': y_test[:10],
    'Scikit-Learn': np.round(y_pred_sklearn[:10], 2),
    'Собственная модель': np.round(y_pred_custom[:10].flatten(), 2)
})
print("\nТаблица предсказаний (первые 10 значений):")
print(results.to_string(index=False))
