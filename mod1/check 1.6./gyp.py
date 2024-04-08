import os
import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind
import numpy as np

# Загрузка данных
current_dir = os.getcwd()
file_path = os.path.join(current_dir, 'churn.csv')
df = pd.read_csv(file_path)

# Предобработка данных
# Удаление дубликатов
df.drop_duplicates(inplace=True)

# Проверка на неявные пропуски в столбце TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df['TotalCharges'].fillna(0, inplace=True)

# Изменение типа данных
df['MonthlyCharges'] = df['MonthlyCharges'].astype(float)
df['TotalCharges'] = df['TotalCharges'].astype(float)
df['tenure'] = df['tenure'].astype(int)

# Проверка гипотезы
# Разделение данных по статусу ухода клиента
churn_yes = df[df['Churn'] == 'Yes']['tenure']
churn_no = df[df['Churn'] == 'No']['tenure']

# T-тест
t_stat, p_value = ttest_ind(churn_yes, churn_no)
print(f't-statistic: {t_stat}, p-value: {p_value}')

# Анализ на нормальность распределения числовых признаков
_, p_value_mc = stats.shapiro(df['MonthlyCharges'].sample(500))  # используем выборку из-за ограничений теста
_, p_value_tc = stats.shapiro(df['TotalCharges'].sample(500))
_, p_value_tenure = stats.shapiro(df['tenure'].sample(500))

print(f'P-value для MonthlyCharges: {p_value_mc}')
print(f'P-value для TotalCharges: {p_value_tc}')
print(f'P-value для tenure: {p_value_tenure}')
