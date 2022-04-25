from ast import Expression

 
import csv
import matplotlib.pyplot as plt
import numpy as np
 

salary_list_training = np.array([])
salary_list_test = np.array([])
height_list_training = np.array([])
height_list_test = np.array([])

salary_list_training2 = np.array([])
salary_list_test2 = np.array([])
height_list_training2 = np.array([])
height_list_test2 = np.array([])

with open('/Users/ugur_dura/Desktop/CE475/Labs/Exercises/CE475-LAB3/Football_players(6).csv', encoding="utf8", errors='ignore') as f:
    data= list(csv.reader(f))

print(data)


for row in data:
    if data[0] != None:
        if row != data[0]:
            if data.index(row) > 25:
                salary_list_training = np.append(salary_list_training, int(row[8]))
                height_list_training = np.append(height_list_training, int(row[5]))

for row in data:
    if data[0] != None:
        if row != data[0]:
            if data.index(row) <= 25:
                salary_list_test = np.append(salary_list_test, int(row[8]))
                height_list_test = np.append(height_list_test, int(row[5]))


for row in data:
    if data[0] != None:
        if row != data[0]:
            if data.index(row) <= 75:
                salary_list_training2 = np.append(salary_list_training2, int(row[8]))
                height_list_training2 = np.append(height_list_training2, int(row[5]))

for row in data:
    if data[0] != None:
        if row != data[0]:
            if data.index(row) > 75:
                salary_list_test2 = np.append(salary_list_test2, int(row[8]))
                height_list_test2 = np.append(height_list_test2, int(row[5]))


print("Training Data: ")
print( salary_list_training)
print(height_list_training)

print("Test Data: ")
print(salary_list_test)
print(height_list_test)

print("Training Size: ")
print("Salary :",salary_list_training.size)
print("Height: ",height_list_training.size)

print("Test Size: ")
print("Salary: ", salary_list_test.size)
print("Height: ", height_list_test.size)

print("Training Data 2: ")
print( salary_list_training2)
print(height_list_training2)

print("Test Data 2: ")
print(salary_list_test2)
print(height_list_test2)

print("Training Size 2: ")
print("Salary :",salary_list_training2.size)
print("Height: ",height_list_training2.size)

print("Test Size 2: ")
print("Salary: ", salary_list_test2.size)
print("Height: ", height_list_test2.size)


def simlin_coef(x,y):
    b1num = 0;
    b1den = 0;

    x_average = np.average(x)
    y_average = np.average(y)

    for i in range (len(x)):
        if len(x) == len(y):
            b1num += (x[i] - x_average) * (y[i] - y_average)
            b1den += (x[i] - x_average) ** 2
    b_one =b1num / b1den
    b_zero = y_average - b_one *x_average

    return b_zero, b_one

b0, b1 = simlin_coef(height_list_training, salary_list_training )
b2, b3 = simlin_coef(height_list_test, salary_list_test)

print("b0 of height training: ",b0)
print("b2 of height test: ",b2)
print("b1 of salary training: ",b1)
print("b3 of salary test: ",b3)


def simling_plot(x, y, b_zero, b_one):
    y_two = b_one * x + b_zero

    plt.figure()
    plt.scatter(x, y, c="b")
    plt.xlabel("Height")
    plt.ylabel("Salary")
    plt.title("Simple Linear Regression: Height vs Salary")

    plt.plot(x, y_two, c="r")
    return y_two


sal_pred_1 = simling_plot(height_list_training, salary_list_training, b2, b3)
sal_pred_2 = simling_plot(height_list_test, salary_list_test, b0, b1)

plt.show()
