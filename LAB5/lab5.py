import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import csv



salary_list_1 = np.array([])
skill_list_1 = np.array([])
 


with open('LAB5/Football_players(11).csv', encoding="utf8", errors='ignore') as f:
    data = list(csv.reader(f))

print(data)


for row in data:
    if row != data[0]:
        salary_list_1 = np.append(salary_list_1, int(row[8]))
        skill_list_1 = np.append(skill_list_1, int(row[7]))
 
print(salary_list_1)
print(skill_list_1)

def calculate_cof(x, y):
    x_transpose = np.transpose(x)
    coef = np.dot(np.dot(np.linalg.inv(np.dot(x_transpose, x)), x_transpose), y)
    return coef

def simlin_coef(x, y):
    b1num = 0
    b1den = 0
    average_x = np.mean(x)
    average_y = np.mean(y)
    for i in range(len(x)):
        b1num += (x[i] - average_x) * (y[i] - average_y)
        b1den += np.square(x[i] - average_x)
    b1 = b1num / b1den
    b0 = average_y - b1 * average_x
    return b0, b1

def mse_func(y, y_est):
    total = 0
    for i in range(len(y)):
       total += (y[i]-y_est[i]) ** 2
    mse = total / len(y)
    return mse

def cubic_sp(x, y, knot_array):
    matrix = np.ones(len(x))
    for i in range(1, 4):
        matrix=np.column_stack((matrix, x ** i))
    for i in range(len(knot_array)):
        col = x - knot_array[i]
        col[col < 0] = 0
        matrix = np.column_stack((matrix, col ** 3))
    matrix = np.column_stack((matrix, y))
    matrix_with_y = matrix[:, 1].argsort()
    matrix = matrix[matrix_with_y]
    matrix_x = matrix[:, :-1] 
    matrix_y = matrix[:, -1]  
    return matrix_x, matrix_y

array = ([25, 50, 75])
array2 = ([40, 85])
array3 = ([90])

matrix_x_1, matrix_y_1 = cubic_sp(skill_list_1, salary_list_1, array)
matrix_x_2, matrix_y_2 = cubic_sp(skill_list_1, salary_list_1, array2)
matrix_x_3, matrix_y_3 = cubic_sp(skill_list_1, salary_list_1, array3)

coefficients_1 = calculate_cof(matrix_x_1, matrix_y_1)
y_pred_1 = np.dot(matrix_x_1, coefficients_1)

coefficients_2 = calculate_cof(matrix_x_2, matrix_y_2)
y_pred_2 = np.dot(matrix_x_2, coefficients_2)

coefficients_3 = calculate_cof(matrix_x_3, matrix_y_3)
y_pred_3 = np.dot(matrix_x_3, coefficients_3)

b0_linear, b1_linear = simlin_coef(skill_list_1, salary_list_1)
y_pred_4 = b1_linear * skill_list_1 + b0_linear

mse_1 = mse_func(matrix_y_1, y_pred_1)
mse_2 = mse_func(matrix_y_2, y_pred_2)
mse_3 = mse_func(matrix_y_3, y_pred_3)
mse_4 = mse_func(salary_list_1, y_pred_4)

print(mse_1)
print(mse_2)
print(mse_3)
print(mse_4)