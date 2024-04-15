import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Загрузка данных
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
data = pd.read_csv(url, header=0, index_col=0, parse_dates=True).squeeze("columns")

# Преобразуем индекс, чтобы был в формате временного ряда с указанием частоты
data.index = pd.date_range(start=data.index[0], periods=len(data), freq='MS')

# Преобразование данных в логарифмический масштаб
data_log = np.log(data)

# Определение параметров модели
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)  # где 12 - период сезонности (месяцы)

# Создание и обучение модели SARIMAX
try:
    model = SARIMAX(data_log,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    initialization='approximate_diffuse')

    results = model.fit(maxiter=200, method='powell', disp=True)

    if not results.mle_retvals['converged']:
        print("Модель не сходится. Попробуйте изменить параметры или метод оптимизации.")
    else:
        print("Модель успешно обучена.")
        print(results.summary())

        # Прогнозирование будущих значений
        forecast = results.get_forecast(steps=24)
        forecast_ci = forecast.conf_int()
        forecast_values = np.exp(forecast.predicted_mean)
        forecast_ci = np.exp(forecast_ci)

        # Визуализация прогноза
        plt.figure(figsize=(10, 5))
        plt.plot(data.index, data, label='Original')
        plt.plot(forecast_values.index, forecast_values, label='Forecast', color='red')
        plt.fill_between(forecast_values.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink')
        plt.title('Forecast vs Actuals')
        plt.legend()
        plt.show()

except Exception as e:
    print("Произошла ошибка при подгонке модели:", e)
