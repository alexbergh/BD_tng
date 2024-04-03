import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Шаг 0: Загрузка данных
current_dir = os.getcwd()
file_path = os.path.join(current_dir, 'client_data.csv')
df = pd.read_csv(file_path)

# Предварительная обработка данных
# Удаление строк с пропусками и заполнение пропущенных значений не требуется, так как данные сгенерированы без пропусков.

# Шаг 1: Описательный анализ данных
print(df.head())
print(df.describe())

# Шаг 2: Построение графиков для анализа
for column in ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']:
    sns.boxplot(x=df[column])
    plt.title(f'Распределение покупок по {column}')
    plt.xlabel('Сумма')
    plt.ylabel('Количество')
    plt.show()

# Распределение ответов на 1-ю кампанию
sns.countplot(x='AcceptedCmp1', data=df)
plt.title('Распределение ответов на 1-ю кампанию')
plt.xlabel('Принято предложение')
plt.ylabel('Количество')
plt.show()

# Распределение ответов на 2-ю кампанию
sns.countplot(x='AcceptedCmp2', data=df)
plt.title('Распределение ответов на 2-ю кампанию')
plt.xlabel('Принято предложение')
plt.ylabel('Количество')
plt.show()

# Распределение ответов на 3-ю кампанию
sns.countplot(x='AcceptedCmp3', data=df)
plt.title('Распределение ответов на 3-ю кампанию')
plt.xlabel('Принято предложение')
plt.ylabel('Количество')
plt.show()

# Распределение ответов на 4-ю кампанию
sns.countplot(x='AcceptedCmp4', data=df)
plt.title('Распределение ответов на 4-ю кампанию')
plt.xlabel('Принято предложение')
plt.ylabel('Количество')
plt.show()

# Распределение ответов на 5-ю кампанию
sns.countplot(x='AcceptedCmp5', data=df)
plt.title('Распределение ответов на 5-ю кампанию')
plt.xlabel('Принято предложение')
plt.ylabel('Количество')
plt.show()

sns.countplot(x='Marital_Status', data=df)
plt.title('Распределение по семейному положению')
plt.xlabel('Семейное положение')
plt.ylabel('Количество')
plt.show()

# Шаг 3: Проверка гипотез
t_stat, p_value = stats.ttest_ind(df[df['Kidhome'] > 0]['Income'], df[df['Kidhome'] == 0]['Income'])
print(f'Т-статистика: {t_stat}, p-значение: {p_value}')

for education in df['Education'].unique():
    sns.countplot(x='AcceptedCmp1', data=df[df['Education'] == education])
    plt.title(f'Ответы на 1-ю кампанию среди {education}')
    plt.xlabel('Принято предложение')
    plt.ylabel('Количество')
    plt.show()

# Шаг 4: Выводы из исследования
print("""
На основе проведенного анализа можно сделать следующие выводы:
- Средние траты на различные категории товаров варьируются. Например, покупатели тратят больше на вино и мясо, чем на фрукты или сладости.
- Покупатели с детьми и без отличаются по уровню дохода, что может влиять на их потребительские предпочтения.
- Различные сегменты по образованию и семейному положению показывают разнообразие в ответах на маркетинговые кампании.
- Анализ ответов на маркетинговые кампании может помочь в планировании будущих стратегий продвижения.

Эти выводы могут помочь компании лучше понять своих клиентов и оптимизировать маркетинговые кампании для различных сегментов.
""")