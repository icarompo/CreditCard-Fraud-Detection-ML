import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

dataset = pd.read_csv('card_transdata.csv') 

X = dataset.drop('fraud', axis=1)  
y = dataset['fraud'] 

pca = PCA(n_components=2) 
X_pca = pca.fit_transform(X)

mask = ((X_pca[:, 0] >= -4000) & (X_pca[:, 0] <= 4000)) & ((X_pca[:, 1] >= -4000) & (X_pca[:, 1] <= 4000))
X_filtered = X_pca[mask]
y_filtered = y[mask]

plt.figure(figsize=(8, 6))
plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_filtered, cmap='coolwarm', marker='o', edgecolor='k')
plt.title('Gráfico de Dispersão Bidimensional (PCA)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')

plt.colorbar(label='Fraude')
plt.show()
