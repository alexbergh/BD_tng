import pandas as pd
import os
from datetime import datetime

# Получаем текущую директорию и формируем путь к файлу
current_dir = os.getcwd()
file_path = os.path.join(current_dir, 'googleplaystore.csv')

# Загрузка DataFrame
df = pd.read_csv(file_path)

# Задача 1: Применение лямбда-функции для категоризации рейтинга
df['categorical_rating'] = df['Rating'].apply(lambda x: 'High rating' if x >= 4.5 else ('Middle rating' if x > 3.8 else 'Low rating'))
print("Результат задачи 1 - категоризация рейтинга:")
print(df[['Rating', 'categorical_rating']].head())

# Задача 2: Применение функции transform
df['mean_cat_rating'] = df.groupby('Category')['Rating'].transform('mean')
print("\nРезультат задачи 2 - средний рейтинг по категориям в новом столбце:")
print(df[['Category', 'Rating', 'mean_cat_rating']].drop_duplicates(subset=['Category']).head())

# Задача 3: Извлечение числа месяца из столбца 'Last Updated'
def extract_month_from_date(date_str):
    try:
        return datetime.strptime(date_str, '%B %d, %Y').month
    except ValueError:
        return 'miss_date'

df['day_of_update'] = df['Last Updated'].apply(extract_month_from_date)
print("\nРезультат задачи 3 - извлечение числа месяца из даты последнего обновления:")
print(df[['Last Updated', 'day_of_update']].head())
