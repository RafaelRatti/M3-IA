import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

data_train = pd.read_csv('bank/bank.csv', delimiter=',')
data_test = pd.read_csv('bank/bank-full.csv', delimiter=',')

data_train = pd.get_dummies(data_train, drop_first=True)
data_test = pd.get_dummies(data_test, drop_first=True)

data_test = data_test.reindex(columns=data_train.columns, fill_value=0)

target_column = "y_yes"
X_train, y_train = data_train.drop(target_column, axis=1), data_train[target_column]
X_test, y_test = data_test.drop(target_column, axis=1), data_test[target_column]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(30, 20, 10), max_iter=10000, tol=1e-9, random_state=42)
mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Matriz de Confusão (Conjunto de Teste):")
print(conf_matrix)
print(f"Acurácia: {accuracy:.4f}")
print(f"Precisão: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Recall: {recall:.4f}")
