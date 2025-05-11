import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def read_data(filename):
    df = pd.read_csv(filename)
    return df


def select_columns(df):
    print("Доступные столбцы:", list(df.columns))
    x_col = input("Введите название столбца для X: ")
    y_col = input("Введите название столбца для Y: ")
    return df[x_col].values, df[y_col].values


def print_statistics(x, y):
    stats = {
        'Количество': [len(x), len(y)],
        'Минимум': [np.min(x), np.min(y)],
        'Максимум': [np.max(x), np.max(y)],
        'Среднее': [np.mean(x), np.mean(y)]
    }
    stat_df = pd.DataFrame(stats, index=['X', 'Y'])
    print("\nСтатистика:")
    print(stat_df)


def plot_data(x, y, m, b, show_squares):
    plt.scatter(x, y, color='blue', label='Исходные данные')

    x_line = np.linspace(np.min(x), np.max(x), 100)
    y_line = m * x_line + b
    plt.plot(x_line, y_line, color='red', label='Регрессионная прямая')

    if show_squares:
        for xi, yi in zip(x, y):
            y_pred = m * xi + b
            plt.vlines(xi, min(yi, y_pred), max(yi, y_pred), color='green', linestyle='--', alpha=0.5)
            plt.fill_between([xi], yi, y_pred, color='green', alpha=0.1)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Линейная регрессия с квадратами ошибок')
    plt.legend()
    plt.grid(True)
    plt.show()


def linear_regression(x, y):
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x ** 2)

    m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    b = (sum_y - m * sum_x) / n
    return m, b


def main():
    filename = "student_scores.csv"
    df = read_data(filename)
    x, y = select_columns(df)
    print_statistics(x, y)
    m, b = linear_regression(x, y)
    print(f"\nУравнение регрессии: y = {m:.2f}x + {b:.2f}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.scatter(x, y, color='blue')
    plt.title('Исходные данные')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.subplot(1, 3, 2)
    plt.scatter(x, y, color='blue')
    x_line = np.linspace(np.min(x), np.max(x), 100)
    plt.plot(x_line, m * x_line + b, color='red')
    plt.title('С регрессионной прямой')
    plt.xlabel('X')
    plt.subplot(1, 3, 3)
    plot_data(x, y, m, b, show_squares=True)
    plt.title('С квадратами ошибок')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()