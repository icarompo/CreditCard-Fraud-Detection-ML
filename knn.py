import pandas as pd
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

print("Acurácia do modelo:", accuracy)
print("\nMatriz de Confusão:\n", confusion)
print("\nRelatório de Classificação:\n", classification_rep)
