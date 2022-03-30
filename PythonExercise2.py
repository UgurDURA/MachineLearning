# -------------------------------------------------------
# [LAB2]
# -------------------------------------------------------
# QUESTION1

import numpy as np
import matplotlib.pyplot as plt

 
matrix = np.random.randint(0,100,(20,4))

print(matrix)
print(matrix.shape)

matrix = np.delete(matrix,0,0)

print("\n First row deleted matrix")
print("\n")

print(matrix)
print(matrix.shape)

matrix = np.delete(matrix,len(matrix)-1,0)

print("\n Last row deleted matrix")
print("\n")

print(matrix)
print(matrix.shape)

matrix = np.delete(matrix,range(1,5),0)
print("\n 2,3,4,5 rows are deleted ")
print("\n")

print(matrix)
print(matrix.shape)

matrix = np.delete(matrix,1,1)
print("\n 2nd column deleted matrix")
print("\n")

print(matrix)
print(matrix.shape)

col = np.linspace(1, 14, 14)

print("\n column of 14 linearly spaced integers")
print("\n")

print(col)
print(col.shape)

matrix = np.column_stack((matrix,col))

print("\n Stacked with column of 14 linearly spaced integers")
print("\n")

print(matrix)
print(matrix.shape)


# -------------------------------------------------------
# QUESTION2 

first_col = matrix[:,0]
last_col = matrix [:,np.size(matrix,1)-1]

print(first_col)
print(last_col)

plt.scatter(last_col, first_col)


y_mean = np.mean(first_col)

plt.plot([0,14], [y_mean, y_mean])
plt.hlines(y_mean, xmin=0, xmax=14)




plt.show()