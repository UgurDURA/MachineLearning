from asyncore import read
import csv
import numpy as np
import matplotlib.pyplot as plt

experience = np.array([])
salary = np.array([])

with open('LAB2/team_big.csv', encoding="utf8", errors='ignore') as f:
    team_big = list(csv.reader(f))

print(team_big)

for row in team_big:
    if row != team_big[0]:
        salary = np.append(salary, int(row[8]))
        experience = np.append(experience, int(row[6]))


print(salary)
print(experience)


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



def simling_plot(x, y, b_zero, b_one):
    y_two = b_one * x + b_zero

    plt.figure()
    plt.scatter(x, y, c="b")
    plt.xlabel("Experience")
    plt.ylabel("Salary")
    plt.title("Simple Linear Regression: Experience vs Salary")

    plt.plot(x, y_two, c="r")
    plt.show()




b_zero, b_one = simlin_coef(experience, salary)

simling_plot(experience, salary, b_zero, b_one)





