import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
current_dir = os.getcwd()
file_path = os.path.join(current_dir, 'diabetes_prediction_dataset.csv')
df = pd.read_csv(file_path)

# 1. Описательный анализ данных
print("\nОписательный анализ данных:")
print(df.describe())
print(df.info())

# 2. Предобработка данных: Удаление дубликатов
print("\nПредобработка данных: Удаление дубликатов")
initial_rows = df.shape[0]
df.drop_duplicates(inplace=True)
print(f"Удалено дубликатов: {initial_rows - df.shape[0]}")

# 3. Предобработка данных: Отсутствующие значения
print("\nОбработка отсутствующих значений:")
print(f"Количество NaN до:\n{df.isna().sum()}")
df.fillna(df.median(numeric_only=True), inplace=True)
for column in df.select_dtypes(include=['object']).columns:
    df[column].fillna(df[column].mode()[0], inplace=True)
print(f"Количество NaN после:\n{df.isna().sum()}")

# 4. Изменение типа данных
df['age'] = df['age'].astype(float)
df['HbA1c_level'] = df['HbA1c_level'].astype(float)

# Визуализация данных
# Гистограммы числовых переменных
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_columns].hist(figsize=(10, 10))
plt.show()

# Диаграммы размаха
for column in numerical_columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[column])
    plt.title(column)
    plt.show()

# Столбчатые диаграммы для категориальных переменных
categorical_columns = df.select_dtypes(include=['object']).columns
for column in categorical_columns:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=df[column])
    plt.title(column)
    plt.show()

# Сравнение выборок: люди с диабетом и без
# Гистограммы для числовых переменных
for column in numerical_columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x=column, hue="diabetes", element="step", stat="density", common_norm=False)
    plt.title(f'Распределение {column} по статусу диабета')
    plt.show()

# Диаграммы размаха для числовых переменных
for column in numerical_columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='diabetes', y=column, data=df)
    plt.title(f'Ящики с усами для {column} по статусу диабета')
    plt.show()

# Матрица корреляции признаков
plt.figure(figsize=(10, 8))
corr_matrix = df.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Матрица корреляции признаков')
plt.show()
