import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Шаг 0: Загрузка данных
current_dir = os.getcwd()
file_path = os.path.join(current_dir, 'marketing_campaign.csv')
df = pd.read_csv(file_path, sep='\t')

# Убедимся, что лишние пробелы в названиях колонок устранены
df.columns = df.columns.str.strip()

# Предварительная обработка данных (пример удаления лишних колонок и проверка пропусков, если потребуется)
# df.drop(['Z_CostContact', 'Z_Revenue'], axis=1, inplace=True)  # Пример удаления неиспользуемых колонок

# Шаг 1: Описательный анализ данных
print(df.head())
print(df.describe())

# Шаг 2: Построение графиков для анализа
columns_to_plot = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
for column in columns_to_plot:
    plt.figure(figsize=(10, 6))  # Установить размер графика
    sns.boxplot(x=df[column])
    plt.title(f'Распределение покупок по {column}')
    plt.xlabel('Сумма')
    plt.show()

# Распределение ответов на маркетинговые кампании и по семейному положению
campaign_columns = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
for column in campaign_columns + ['Marital_Status']:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=column, data=df)
    plt.title(f'Распределение по {column}')
    plt.ylabel('Количество')
    plt.show()

# Шаг 3: Проверка гипотез
# Проверка различий в доходах между семьями с детьми и без
t_stat, p_value = stats.ttest_ind(df[df['Kidhome'] > 0]['Income'], df[df['Kidhome'] == 0]['Income'], nan_policy='omit')  # Учитываем NaN значения
print(f'Т-статистика: {t_stat}, p-значение: {p_value}')

# Анализ влияния образования на принятие предложений первой кампании
for education in df['Education'].unique():
    plt.figure(figsize=(10, 6))
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