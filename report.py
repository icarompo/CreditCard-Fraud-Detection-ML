import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from naive_plot import naive_plot
from tree_plot import tree_plot
from knn_plot import knn_plot

dataset = pd.read_csv('card_transdata.csv')

X = dataset.drop('fraud', axis=1)
y = dataset['fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X_train, y_train)
y_pred_nb = naive_bayes_model.predict(X_test)

decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, y_train)
y_pred_tree = decision_tree_model.predict(X_test)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

accuracy_nb = accuracy_score(y_test, y_pred_nb)
confusion_nb = confusion_matrix(y_test, y_pred_nb)
classification_rep_nb = classification_report(y_test, y_pred_nb)

accuracy_tree = accuracy_score(y_test, y_pred_tree)
confusion_tree = confusion_matrix(y_test, y_pred_tree)
classification_rep_tree = classification_report(y_test, y_pred_tree)

accuracy_knn = accuracy_score(y_test, y_pred_knn)
confusion_knn = confusion_matrix(y_test, y_pred_knn)
classification_rep_knn = classification_report(y_test, y_pred_knn)

def print_confusion_matrix_with_labels(confusion_matrix):
    tp, fp, fn, tn = confusion_matrix.ravel()
    print("\nMatriz de Confusão:")
    print(f"\tVerdadeiros Positivos (TP): {tp}")
    print(f"\tFalsos Positivos (FP): {fp}")
    print(f"\tFalsos Negativos (FN): {fn}")
    print(f"\tVerdadeiros Negativos (TN): {tn}")

print("Resultados para Naive Bayes:")
print_confusion_matrix_with_labels(confusion_nb)
print("\nRelatório de Classificação:\n", classification_rep_nb)
print("\nAcurácia do modelo:", accuracy_nb)

print("\n\nResultados para Árvore de Decisão:")
print_confusion_matrix_with_labels(confusion_tree)
print("\nRelatório de Classificação:\n", classification_rep_tree)
print("\nAcurácia do modelo:", accuracy_tree)

print("\n\nResultados para KNN:")
print_confusion_matrix_with_labels(confusion_knn)
print("\nRelatório de Classificação:\n", classification_rep_knn)
print("\nAcurácia do modelo:", accuracy_knn)

#naive_plot()
#tree_plot()
#knn_plot()
