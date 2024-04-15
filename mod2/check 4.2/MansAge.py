import pandas as pd
from sqlalchemy import create_engine, text

# Чтение данных из CSV файла
df = pd.read_csv('Mall_Customers.csv')

# Переименование колонок для удобства обращения в SQL (удаление пробелов)
df.columns = [c.replace(' ', '_') for c in df.columns]

# Создание SQLite in-memory базы данных
engine = create_engine('sqlite://', echo=False)

# Загрузка DataFrame в SQL таблицу
df.to_sql('customers', con=engine, index=False, if_exists='replace')

# Создание SQL запроса для выбора всех мужчин с иерархической сортировкой
query = text("""
SELECT *
FROM customers
WHERE Genre = 'Male'
ORDER BY Age ASC, Annual_Income DESC, Spending_Score ASC;
""")

# Выполнение SQL запроса и вывод результатов
with engine.connect() as connection:
    result = connection.execute(query).fetchall()
    for row in result:
        print(row)
