import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

dataset = pd.read_csv('card_transdata.csv')

X = dataset.drop('fraud', axis=1)
y = dataset['fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, y_train)

y_pred = decision_tree_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("\nMatriz de Confusão:\n", confusion)
print("\nRelatório de Classificação:\n", classification_rep)
print("Acurácia do modelo:", accuracy)

plt.figure(figsize=(12, 8))
plot_tree(decision_tree_model, filled=True, feature_names=X.columns.tolist(), class_names=['Não Fraudulento', 'Fraudulento'], fontsize=8)
plt.title('Árvore de Decisão')
plt.show()
