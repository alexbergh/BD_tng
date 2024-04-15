import pandas as pd
from sqlalchemy import create_engine, text

# Чтение данных из CSV файла
df = pd.read_csv('world-data-2023.csv')

# Создание SQLite in-memory базы данных
engine = create_engine('sqlite://', echo=False)

# Загрузка DataFrame в SQL таблицу
df.to_sql('world_data', con=engine, index=False, if_exists='replace')

# Создание соединения с базой данных и выполнение SQL запроса
with engine.connect() as connection:
    result = connection.execute(text("""
        SELECT Country, Abbreviation, "Life expectancy"
        FROM world_data
        WHERE Abbreviation LIKE 'S%' OR "Life expectancy" < 60
    """)).fetchall()

# Вывод результатов
for row in result:
    print(row)
