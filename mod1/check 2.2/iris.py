import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Загрузка данных
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Преобразование целевой переменной
y = np.where(y == 2, 1, 0)  # 1 для Ирис Вирджиника, иначе 0

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=100)

# Создание и обучение модели логистической регрессии
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Предсказания модели
y_pred = model.predict(X_test)

# Оценка модели
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Оценка качества модели логистической регрессии:")
print(f"Точность: {accuracy:.4f}")
print(f"Точность по классам: {precision:.4f}")
print(f"Полнота: {recall:.4f}")
print(f"F1-мера: {f1:.4f}\n")

# Визуализация матрицы ошибок
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Прогнозируемые значения')
plt.ylabel('Истинные значения')
plt.title('Матрица ошибок')
plt.show()

# Визуализация ROC-кривой
y_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Доля ложноположительных результатов')
plt.ylabel('Доля истинноположительных результатов')
plt.title('Характеристика работы приёмника (ROC-кривая)')
plt.legend(loc="lower right")
plt.show()
