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


# -------------------------------------------------------
# QUESTION3 

y_mean = np.mean(first_col)

plt.plot([0,14], [y_mean, y_mean])

# plt.hlines(y_mean, xmin=0, xmax=14) same with above


# -------------------------------------------------------
# QUESTION4 

# The mean square error (MSE) provides a statistic that allows for researchers 
# to make such claims. MSE simply refers to the mean of the squared difference 
# between the predicted parameter and the observed parameter.
#  
# To find the MSE, take the observed value, subtract the predicted value, and square
#  that difference. Repeat that for all observations. Then, sum all of those squared 
# values and divide by the number of observations.

mse = np.mean(np.square(first_col - y_mean))
print("Mean Square Error : ",mse)
plt.show()