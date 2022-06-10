import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv('players_km.csv')
age_x = data.to_numpy()[:, 4]

mental_x = data.to_numpy()[:, 6]

salary_y = data.to_numpy()[:, 8]

combined_matrix = np.column_stack((age_x, mental_x))
matrix = np.array(combined_matrix, dtype=np.int)

kmeans = KMeans(n_clusters=4).fit(matrix)
labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

matrix1 = np.zeros((1, 2))
matrix2 = np.zeros((1, 2))
matrix3 = np.zeros((1, 2))
matrix4 = np.zeros((1, 2))

for i in range(len(matrix)):
    if(kmeans.labels_[i] == 0):
        matrix1 = np.vstack([matrix1, matrix[i, :]])
    if(kmeans.labels_[i] == 1):
        matrix2 = np.vstack([matrix2, matrix[i, :]])
    if (kmeans.labels_[i] == 2):
        matrix3 = np.vstack([matrix3, matrix[i, :]])
    if (kmeans.labels_[i] == 3):
        matrix4 = np.vstack([matrix4, matrix[i, :]])

print(labels)
matrix1 = np.delete(matrix1, 0, axis=0)
matrix2 = np.delete(matrix2, 0, axis=0)
matrix3 = np.delete(matrix3, 0, axis=0)
matrix4 = np.delete(matrix4, 0, axis=0)
print(matrix)
print(matrix1)
print(matrix2)
print(matrix3)
print(matrix4)

plt.figure()
plt.scatter(matrix1[:, 0], matrix1[:, 1])
plt.scatter(matrix2[:, 0], matrix2[:, 1])
plt.scatter(matrix3[:, 0], matrix3[:, 1])
plt.scatter(matrix4[:, 0], matrix4[:, 1])
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], color="none", edgecolor="black", s=250)
plt.show()