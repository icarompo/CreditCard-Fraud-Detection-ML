import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

dataset = pd.read_csv('card_transdata.csv')

X = dataset.drop('fraud', axis=1)
y = dataset['fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("\nMatriz de Confusão:\n", confusion)
print("\nRelatório de Classificação:\n", classification_rep)
print("Acurácia do modelo:", accuracy)

plt.figure(figsize=(8, 6))
plt.scatter(X_test['distance_from_home'], X_test['distance_from_last_transaction'], c=y_pred, cmap='coolwarm', marker='o', edgecolor='k')
plt.title('Gráfico de Dispersão - Resultados do KNN')
plt.xlabel('Distância do Domicílio')
plt.ylabel('Distância da Última Transação')

plt.colorbar(label='Classificação (0 - Não Fraudulento, 1 - Fraudulento)')
plt.show()
