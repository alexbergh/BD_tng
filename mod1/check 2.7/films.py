import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score
import random

print("Шаг 1: Загрузка данных")
data = pd.read_csv('ratings_df_sample_2.csv', usecols=['userId', 'movieId', 'rating'])
print("Данные загружены.")

data['watched'] = 1
print("Положительные метки добавлены.")

print("Шаг 2: Генерация 'ноликов' для непросмотренных фильмов.")
user_ids = data['userId'].unique()
movie_ids = data['movieId'].unique()

np.random.seed(42)
random_users = np.random.choice(user_ids, size=2 * len(data), replace=True)
random_movies = np.random.choice(movie_ids, size=2 * len(data), replace=True)
random_data = pd.DataFrame({'userId': random_users, 'movieId': random_movies}).copy()
print("Случайные пары сгенерированы.")

random_data = random_data.drop_duplicates().reset_index(drop=True)
print("Дубликаты удалены.")

existing_pairs = set(zip(data['userId'], data['movieId']))
random_data['pair'] = list(zip(random_data['userId'], random_data['movieId']))
random_data = random_data[~random_data['pair'].isin(existing_pairs)]
print("Отфильтрованы пары, уже присутствующие в данных.")

random_data['watched'] = 0
random_data = random_data.drop('pair', axis=1)
print("Флаг 'watched' установлен для непросмотренных фильмов.")

full_data = pd.concat([data, random_data], ignore_index=True)
print("Данные объединены.")

print("Шаг 3: Подготовка данных к обучению")
train_data, test_data = train_test_split(full_data, test_size=0.2, random_state=42)
print("Данные разделены на тренировочные и тестовые.")

train_matrix = train_data.pivot_table(index='userId', columns='movieId', values='watched', fill_value=0)
train_matrix_csr = csr_matrix(train_matrix.values)
print("Матрица пользователь-фильм создана.")

print("Шаг 4: SVD факторизация")
svd = TruncatedSVD(n_components=50, n_iter=7, random_state=42)
svd.fit(train_matrix_csr)
print("SVD факторизация завершена.")

print("Шаг 5: Обучение и оценка dummy-model")
def dummy_model(n):
    return np.random.rand(n)

dummy_predictions = dummy_model(len(test_data))
roc_auc = roc_auc_score(test_data['watched'], dummy_predictions)
print(f"ROC AUC score for dummy model: {roc_auc:.2f}")

# Визуализация предсказаний dummy model
plt.figure(figsize=(10, 5))
plt.scatter(test_data['watched'], dummy_predictions, alpha=0.2)
plt.title('Сравнение предсказанных вероятностей и реальных данных')
plt.xlabel('Реальные значения')
plt.ylabel('Предсказанные вероятности')
plt.show()

# Визуализация взаимодействий пользователей
user_interactions = full_data.groupby('userId').size()
plt.figure(figsize=(10, 5))
user_interactions.hist(bins=30, alpha=0.7, color='blue')
plt.title('Распределение количества взаимодействий пользователей с системой')
plt.xlabel('Количество взаимодействий')
plt.ylabel('Количество пользователей')
plt.show()

# Функция для рекомендаций
def recommend_movies(user_id, num_recommendations=5):
    user_index = train_matrix.index.get_loc(user_id)
    user_ratings = svd.transform(train_matrix_csr[user_index])
    predicted_ratings = np.dot(user_ratings, movie_features.T)
    recommended_movie_ids = np.argsort(predicted_ratings)[0][-num_recommendations:][::-1]
    recommended_movies = train_matrix.columns[recommended_movie_ids].tolist()
    return recommended_movies

sample_user = np.random.choice(train_matrix.index)
recommendations = recommend_movies(sample_user)
print(f"Рекомендации для пользователя {sample_user}: {recommendations}")
