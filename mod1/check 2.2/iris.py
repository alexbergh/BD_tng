import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load data
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Conversion of the target variable
y = np.where(y == 2, 1, 0)  #1 for Iris Virginica, otherwise 0

# Separation into training and test samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=100)

# Creating and training a logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Model predictions
y_pred = model.predict(X_test)

# Evaluation of the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Evaluation of the quality of the logistic regression model:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy by class: {precision:.4f}")
print(f"Completeness: {recall:.4f}")
print(f"F1 is a measure: {f1:.4f}\n")

# Visualization of the error matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted values')
plt.ylabel('True values')
plt.title('Error matrix')
plt.show()

# Visualization of the ROC curve
y_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Proportion of false positive results')
plt.ylabel('Proportion of true positive results')
plt.title('Receiver performance characteristic (ROC curve)')
plt.legend(loc="lower right")
plt.show()
