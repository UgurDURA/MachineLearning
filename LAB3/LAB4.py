import csv
import matplotlib.pyplot as plt
import numpy as np

salary_list_1 = np.array([])
salary_list_2 = np.array([])
mental_list_1 = np.array([])
mental_list_2 = np.array([])


with open('Football_players(4).csv', encoding="utf8", errors='ignore') as f:
    data = list(csv.reader(f))

print(data)

for row in data:
    if row != data[0]:
        if data.index(row) <= 80:
            salary_list_1 = np.append(salary_list_1, int(row[8]))
            mental_list_1 = np.append(mental_list_1, int(row[6]))
for row in data:
    if row != data[0]:
        if data.index(row) > 20:
            salary_list_1 = np.append(salary_list_2, int(row[8]))
            mental_list_1 = np.append(mental_list_2, int(row[6]))


print(salary_list_1)
print(salary_list_2)

print(mental_list_1)
print(mental_list_2)

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

b0, b1 = simlin_coef(mental_list_1 , salary_list_1)

b2, b3= simlin_coef(mental_list_2, salary_list_2)

print("b0_1 is ", b0)
print("b1_1 is ", b1)
print("b0_2 is ", b2)
print("b1_2 is ", b3)

def simling_plot(x, y, b_zero, b_one):
    y_two = b_one * x + b_zero

    plt.figure()
    plt.scatter(x, y, c="b")
    plt.xlabel("Experience")
    plt.ylabel("Salary")
    plt.title("Simple Linear Regression: Experience vs Salary")

    plt.plot(x, y_two, c="r")
    return y_two

sal_pred_1 = simling_plot(mental_list_1, salary_list_1, b0, b1)
sal_pred_2 = simling_plot(mental_list_2, salary_list_2, b2, b3)

plt.show()