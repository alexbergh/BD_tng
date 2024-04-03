import pandas as pd
import os

# Получаем текущую директорию и формируем путь к файлу
current_dir = os.getcwd()
file_path = os.path.join(current_dir, 'googleplaystore.csv')

# 1. Загрузка DataFrame и вывод первых пяти строк
df = pd.read_csv(file_path)
print(df.head())

# 2. Количество строк и столбцов в DataFrame
print("\nКоличество строк и столбцов:", df.shape)

# 3. Основная информация о DataFrame и анализ пропусков
df.info()

# 4. Удаление или замена пропусков
# Здесь мы предполагаем, что столбец 'Rating' имеет числовые пропуски, а 'Genres' - категориальные
df['Rating'].fillna(df['Rating'].median(), inplace=True)
df.dropna(subset=['Genres'], inplace=True)

# 5. Проверка на пропуски после обработки
print("\nКоличество пропусков после обработки:")
print(df.isnull().sum())

# 6. Минимум и максимум из столбца Price
# Убедимся, что столбец 'Price' уже в числовом формате
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
print("\nМинимальная цена:", df['Price'].min())
print("Максимальная цена:", df['Price'].max())

# 7. Медиана и среднее арифметическое для Rating и Reviews
# Убеждаемся, что 'Reviews' уже в целочисленном формате
df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce').fillna(0).astype(int)
print("\nМедиана рейтинга:", df['Rating'].median())
print("Средний рейтинг:", df['Rating'].mean())
print("Медиана отзывов:", df['Reviews'].median())
print("Среднее количество отзывов:", df['Reviews'].mean())

# 8. Уникальные значения в Genres
print("\nУникальные значения в Genres:")
print(df['Genres'].unique())

# 9. Сгруппировать по Genres и посчитать средний и медианный рейтинг
grouped_df = df.groupby('Genres')['Rating'].agg(['mean', 'median']).reset_index()
print("\nСредний и медианный рейтинг по жанрам:")
print(grouped_df)

# 10. Жанры с медианным рейтингом больше 4.5
print("\nЖанры с медианным рейтингом больше 4.5:")
print(grouped_df[grouped_df['median'] > 4.5])

# 11. Удаление дубликатов
initial_duplicates = df.duplicated().sum()
df.drop_duplicates(inplace=True)
print("\nКоличество дубликатов до удаления:", initial_duplicates)
print("Количество дубликатов после удаления:", df.duplicated().sum())

# 12. Количество приложений в каждом жанре
print("\nКоличество приложений в каждом жанре (ТОП-10):")
print(df['Genres'].value_counts().head(10))
