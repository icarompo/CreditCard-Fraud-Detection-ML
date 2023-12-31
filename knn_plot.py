import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def knn_plot():
    dataset = pd.read_csv('card_transdata.csv')

    X = dataset.drop('fraud', axis=1)
    y = dataset['fraud']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)

    y_pred = knn_model.predict(X_test)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_test['distance_from_home'], X_test['distance_from_last_transaction'], c=y_pred, cmap='coolwarm', marker='o', edgecolor='k')
    plt.title('KNN Results')
    plt.xlabel('Distance from home')
    plt.ylabel('Distance from last transaction')

    plt.colorbar(label='Classification (0 - Non Fraud, 1 - Fraud)')
    plt.show()

knn_plot()