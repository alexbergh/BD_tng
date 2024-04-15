import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

# Загрузка данных
data = pd.read_csv('diamonds.csv')

# Проверка на пропущенные значения
print(data.isnull().sum())

# Описательный анализ данных
print(data.describe())

# Визуализация распределения цены
plt.figure(figsize=(10, 6))
sns.histplot(data['price'], kde=True)
plt.title('Распределение цены')
plt.xlabel('Цена')
plt.ylabel('Количество')
plt.savefig('price_distribution.pdf') # Сохранение графика в PDF
plt.show()

# Визуализация влияния качества огранки на цену до преобразования переменных
plt.figure(figsize=(10, 6))
sns.boxplot(x='cut', y='price', data=data)
plt.title('Влияние огранки на цену')
plt.xlabel('Огранка')
plt.ylabel('Цена')
plt.savefig('price_by_cut.pdf') # Сохранение графика в PDF
plt.show()

# Преобразование категориальных данных с использованием OneHotEncoder
encoder = OneHotEncoder(drop='first')
encoded_features = encoder.fit_transform(data[['cut', 'color', 'clarity']])
encoded_features_df = pd.DataFrame(encoded_features.toarray(), columns=encoder.get_feature_names_out(input_features=['cut', 'color', 'clarity']))
data_encoded = pd.concat([data.drop(['cut', 'color', 'clarity'], axis=1), encoded_features_df], axis=1)

# Визуализация зависимости цены от каратности
plt.figure(figsize=(10, 6))
sns.scatterplot(x='carat', y='price', data=data_encoded)
plt.title('Зависимость цены от каратности')
plt.xlabel('Карат')
plt.ylabel('Цена')
plt.savefig('price_by_carat.pdf') # Сохранение графика в PDF
plt.show()

# Разделение данных на обучающую, валидационную и тестовую выборки
X = data_encoded.drop('price', axis=1)
y = data['price']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Настройка параметров моделей с использованием GridSearchCV
param_grid_lr = {'fit_intercept': [True, False]}
param_grid_rf = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}

grid_search_lr = GridSearchCV(LinearRegression(), param_grid_lr, cv=5, scoring='neg_mean_squared_error')
grid_search_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=5, scoring='neg_mean_squared_error')

grid_search_lr.fit(X_train, y_train)
grid_search_rf.fit(X_train, y_train)

# Выбор лучшей модели на основе результатов GridSearchCV
if grid_search_lr.best_score_ < grid_search_rf.best_score_:
    best_model = grid_search_lr.best_estimator_
    print("Лучшая модель: Линейная регрессия")
else:
    best_model = grid_search_rf.best_estimator_
    print("Лучшая модель: Случайный лес")

# Оценка лучшей модели на тестовой выборке
y_test_pred = best_model.predict(X_test)
print("Тестовое MSE:", mean_squared_error(y_test, y_test_pred))
