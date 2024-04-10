import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Загрузка данных
current_dir = os.getcwd()
file_path = os.path.join(current_dir, 'marketing_campaign.csv')
df = pd.read_csv(file_path, sep='\t')

# Предварительная обработка
df.columns = df.columns.str.strip()
df.dropna(subset=['Income'], inplace=True)  # Удаление записей без данных о доходе

# Добавление общей суммы трат
df['Total_Spend'] = df[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)

# Разделение данных по медиане дохода
median_income = df['Income'].median()
df['Income_Level'] = np.where(df['Income'] >= median_income, 'High', 'Low')

# Разделение данных по медиане трат
median_spend = df['Total_Spend'].median()
df['Spend_Level'] = np.where(df['Total_Spend'] >= median_spend, 'High', 'Low')

# Анализ
print("Средние значения по категориям трат для разных уровней дохода:")
print(df.groupby('Income_Level').mean(numeric_only=True)[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']])

print("\nСредние значения по категориям трат для разных уровней трат:")
print(df.groupby('Spend_Level').mean(numeric_only=True)[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']])

# Корреляция между тратами и доходом
corr_matrix = df[['Income', 'Total_Spend']].corr()
print("\nКорреляция между доходом и общими тратами:")
print(corr_matrix)

# Визуализация
plt.figure(figsize=(10, 6))
sns.heatmap(df[['Income', 'Total_Spend', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].corr(), annot=True, fmt=".2f")
plt.title("Корреляционная матрица")
plt.show()

# Дополнительные анализы
# Можно добавить более сложные анализы, например, кластеризацию или предсказательные модели

### Выводы
# Здесь вы можете добавить свои выводы на основе проведенного анализа
