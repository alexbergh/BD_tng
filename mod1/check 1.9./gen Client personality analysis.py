import pandas as pd
import numpy as np
import os

# Генерация случайных данных для атрибутов
data = {
    'ID': np.random.randint(1000, 9999, 1000), # Уникальный идентификатор клиента
    'Year_Birth': np.random.randint(1950, 2005, 1000), # Год рождения клиента
    'Education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 1000), # Уровень образования клиента
    'Marital_Status': np.random.choice(['Single', 'Married', 'Divorced'], 1000), # Семейное положение клиента
    'Income': np.random.randint(30000, 150000, 1000), # Годовой доход семьи клиента
    'Kidhome': np.random.randint(0, 4, 1000), # Количество детей в семье клиента
    'Teenhome': np.random.randint(0, 3, 1000), # Количество подростков в домохозяйстве клиента
    'Dt_Customer': pd.date_range(start="2020-01-01", end="2023-12-31", periods=1000), # Дата регистрации клиента в компании
    'Recency': np.random.randint(0, 365, 1000), # Количество дней с момента последней покупки клиента
    'Complain': np.random.randint(0, 2, 1000), # 1, если клиент жаловался в течение последних 2 лет, 0 — в противном случае
    'MntWines': np.random.randint(0, 1000, 1000), # Сумма, потраченная на вино за последние 2 года
    'MntFruits': np.random.randint(0, 1000, 1000), # Сумма, потраченная на фрукты за последние 2 года
    'MntMeatProducts': np.random.randint(0, 1000, 1000), # Сумма, потраченная на мясо за последние 2 года
    'MntFishProducts': np.random.randint(0, 1000, 1000), # Сумма, потраченная на рыбу за последние 2 года
    'MntSweetProducts': np.random.randint(0, 1000, 1000), # Сумма, потраченная на сладости за последние 2 года
    'MntGoldProds': np.random.randint(0, 1000, 1000), # Сумма, потраченная на золото за последние 2 года
    'NumDealsPurchases': np.random.randint(0, 10, 1000), # Количество покупок, совершенных со скидкой
    'AcceptedCmp1': np.random.randint(0, 2, 1000), # 1, если клиент принял предложение в 1-й кампании, 0 — иначе
    'AcceptedCmp2': np.random.randint(0, 2, 1000), # 1, если клиент принял предложение во 2-й кампании, 0 — иначе
    'AcceptedCmp3': np.random.randint(0, 2, 1000), # 1, если клиент принял предложение в 3-й кампании, 0 — иначе
    'AcceptedCmp4': np.random.randint(0, 2, 1000), # 1, если клиент принял предложение в 4-й кампании, 0 — иначе
    'AcceptedCmp5': np.random.randint(0, 2, 1000), # 1, если клиент принял предложение в 5-й кампании, 0 — иначе
    'Response': np.random.randint(0, 2, 1000), # 1, если клиент принял предложение в последней кампании, 0 — иначе
    'NumWebPurchases': np.random.randint(0, 10, 1000), # Количество покупок, совершенных через веб-сайт компании
    'NumCatalogPurchases': np.random.randint(0, 10, 1000), # Количество покупок, сделанных с помощью каталога
    'NumStorePurchases': np.random.randint(0, 10, 1000), # Количество покупок, сделанных непосредственно в магазинах
    'NumWebVisitsMonth': np.random.randint(0, 31, 1000) # Количество посещений веб-сайта компании за последний месяц
}

# Создание DataFrame
df = pd.DataFrame(data)

# Сохранение DataFrame в CSV файл
current_dir = os.getcwd()
df.to_csv(os.path.join(current_dir, "client_data.csv"), index=False)
print("Файл client_data.csv успешно создан.")