
from sre_constants import AT_MULTILINE
import numpy as np

# QUESTION1
# -------------------------------------------------------
arr = np.array([])
arr = np.random.randint(-100,100,25)
     
print(arr)

evens = 0
odds = 0

for elem in arr:
    if elem % 2 == 0:
        evens +=1
    else:
        odds +=1


print("Evens: ",evens)
print("Odds: "+str(odds))

# -------------------------------------------------------
# QUESTION2


# sum_arr = 0

# for i in range(len(arr)):
#      sum_arr += arr[i]

# avg_arr = summ_arr/len(arr)

sum_arr = np.sum(arr)
avg_arr = np.mean(arr)


print("Sum: ",sum_arr)
print("Average:", avg_arr)

# -------------------------------------------------------
# QUESTION3

# matrix = np.zeros((3,3))
# # len(matrix) number of rows

# num_rows = np.size(matrix, 0)
# num_cols = np.size(matrix, 1)
# matrix_array = np.random.randint(0,2,(3,3))

matrix = np.random.random((3,3))
print(matrix)

print("\n Second Row: ")
print(matrix[1])

print("\n Third Column: ")
print(matrix[:, 2])

print("\n Second & Third rows: ")
print(matrix[1:3])

print("\n 1st & 2nd rows; 2nd & 3rd Columns: ")
print(matrix[0:2,1:3])

print("\nTranspose of combined second & third rows: ")
# print(np.transpose(matrix[1:3]))
print(matrix[1:3].T)

# -------------------------------------------------------
# QUESTION4

a = np.linspace(1,6,10,endpoint=False)
b = np.arange(6,1,-.5)

print("\nLinaerly Spaced ascending numbers :")
print(a)
print("\nLinaerly Spaced decending numbers :")
print(b)

 
v_stacked = np.vstack((a,b))
print("\nvertically stacked matrixes: ")
print(v_stacked)

h_stacked = np.column_stack((a,b))
print("\nHorizontally stacked matrixes :")
print(h_stacked)

result = 0

for i in range(len(a)):
    result += (a[i]-b[i]) ** 2
result /= len(a)

print("\nThe result of Substriction between two matrixes: ")
print(result)

result2 = np.mean(np.square(a-b))
print("\nThe result of Substriction between two matrixes with np functions: ")
print(result2)


