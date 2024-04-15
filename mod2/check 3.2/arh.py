import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import AUC

# Загрузка данных
data = pd.read_csv('train_3.2.csv')

# Основной анализ данных
print(data.head())
print(data.describe())
print(data.isnull().sum())
print(data['defects'].value_counts())

# Корреляционная матрица
plt.figure(figsize=(12, 10))
sns.heatmap(data.corr(), annot=True, fmt=".2f")
plt.show()

# Предобработка данных
X = data.drop(['id', 'defects'], axis=1)
y = data['defects'].astype(np.float32)

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Нормализация данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Функция построения базовой модели
def build_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[AUC(name='auc')])
    return model

# Обучение модели
model = build_model()
history = model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=100, batch_size=32)

# Построение модели с регуляризацией
def build_regularized_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[AUC(name='auc')])
    return model

# Коллбэки
early_stopping = EarlyStopping(monitor='val_auc', patience=10, mode='max', restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model', monitor='val_auc', save_best_only=True, mode='max')

# Обучение модели с регуляризацией
model_reg = build_regularized_model()
history_reg = model_reg.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=100, batch_size=32, callbacks=[early_stopping, model_checkpoint])

# Загрузка и оценка лучшей модели
best_model = tf.keras.models.load_model('best_model')
_, test_auc = best_model.evaluate(X_test_scaled, y_test)
print(f'Test AUC: {test_auc}')

# Визуализация результатов
plt.plot(history.history['auc'], label='Training AUC (Base Model)')
plt.plot(history.history['val_auc'], label='Validation AUC (Base Model)')
plt.plot(history_reg.history['auc'], label='Training AUC (Regularized Model)')
plt.plot(history_reg.history['val_auc'], label='Validation AUC (Regularized Model)')
plt.title('Model AUC Performance')
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.legend()
plt.show()
