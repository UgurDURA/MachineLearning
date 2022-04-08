from ast import Expression
import csv
import matplotlib.pyplot as plt
import numpy as np

salary_list_1 = np.array([])
salary_list_2 = np.array([])
exp_list_1 = np.array([])
exp_list_2 = np.array([])

with open('LAB2/team_big.csv', encoding="utf8", errors='ignore') as f:
    team_big = list(csv.reader(f))

for row in team_big:
    if row != team_big[0]:
        if team_big.index(row) <= 20:
            salary_list_1 = np.append(salary_list_1, int(row[8]))
            exp_list_1 = np.append(exp_list_1, int(row[6]))

for row in team_big:
    if row != team_big[0]:
        if team_big.index(row) > 20:
            salary_list_2 = np.append(salary_list_2, int(row[8]))
            exp_list_2 = np.append(exp_list_2, int(row[6]))


print(salary_list_1)
print(exp_list_1)
print(salary_list_2)
print(exp_list_2)

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


b0_1, b1_1 = simlin_coef(exp_list_1, salary_list_1)
b0_2, b1_2 = simlin_coef(exp_list_2, salary_list_2)

print("b0_1 is ", b0_1)
print("b1_1 is ", b1_1)
print("b0_2 is ", b0_2)
print("b1_2 is ", b1_2)

def simling_plot(x, y, b_zero, b_one):
    y_two = b_one * x + b_zero

    plt.figure()
    plt.scatter(x, y, c="b")
    plt.xlabel("Experience")
    plt.ylabel("Salary")
    plt.title("Simple Linear Regression: Experience vs Salary")

    plt.plot(x, y_two, c="r")
    return y_two


sal_pred_1 = simling_plot(exp_list_1, salary_list_1, b0_2, b1_2)
sal_pred_2 = simling_plot(exp_list_2, salary_list_2, b0_1, b1_1)

def simlin_calculate (a_y, e_y):
    rss = 0
    tss = 0
    average_y = np.average(a_y)

    for i in range(len(a_y)):
        if len(a_y) == len (e_y):
            rss += (a_y[i] - e_y[i]) **2
            tss += (a_y[i] - average_y) ** 2

    r_square = 1 - (rss / tss)

    return r_square



r_square_1 = simlin_calculate(salary_list_1, sal_pred_1)
r_square_2 = simlin_calculate(salary_list_2, sal_pred_2)

print("R^2 score: ", r_square_1)
print("R^2 score: ", r_square_2)

plt.show()