import pandas as pd
import os

# Получаем текущую директорию и формируем путь к файлу
current_dir = os.getcwd()
file_path = os.path.join(current_dir, 'googleplaystore.csv')

# Загрузка DataFrame
df = pd.read_csv(file_path)

# Предобработка данных
# 1. Изменение названий всех столбцов, приведя их к нижнему регистру
df.columns = df.columns.str.lower()

# 2. Вывод основных статистик по категориальным столбцам
print("\n#2 Основные статистики по категориальным столбцам:")
print(df.describe(include=['object']))

# Преобразование столбца `rating` из строк в числа для выполнения числовых операций
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

# 3. Вывод части датафрейма для приложений из категории "Art & Design" с рейтингом больше 4.5
print("\n#3 Приложения из категории 'Art & Design' с рейтингом больше 4.5:")
filtered_df = df[(df['category'].str.upper() == 'ART_AND_DESIGN') & (df['rating'] > 4.5)]
print(filtered_df)

# 4. Количество игр в рейтинге с разбиением 100 и вывод топ-10 игр по количеству в рейтинге
games_df = df[df['category'].str.upper() == 'GAME']
rating_counts = games_df['rating'].value_counts(bins=100).sort_values(ascending=False).head(10)
print("\n#4 Топ-10 игр по количеству в рейтинге:")
print(rating_counts)

# 5. Применение функции для категоризации рейтинга
def categorize_rating(rating):
    if rating >= 4.5:
        return 'High rating'
    elif rating > 3.8:  # Исправлено условие для соответствия условию задачи
        return 'Middle rating'
    else:
        return 'Low rating'

df['categorical_rating'] = df['rating'].apply(categorize_rating)
print("\n#5 Добавлен новый столбец с категоризированным рейтингом:")

# 6. Вывод количества уникальных значений в столбцах, отсортированных по убыванию
print("\n#6 Количество уникальных значений в столбцах, отсортированных по убыванию:")
print(df.nunique().sort_values(ascending=False))
