import csv
from re import X
from matplotlib import pyplot as plt
import numpy as np 

x1_AgeList =np.array([])
x2_HeightList = np.array([])
x3_MentalStrengthList = np.array([])
x4_SkillList = np.array([])

with open('CE475-LAB3/Football_players(6).csv', encoding="utf8", errors='ignore') as f:
    data = list(csv.reader(f))

print(data)

for row in data:
    if row != data[0]:
        x1_AgeList = np.append(x1_AgeList, int(row[4]))
        x2_HeightList =np.append(x2_HeightList, int(row[5]))
        x3_MentalStrengthList = np.append( x3_MentalStrengthList, int(row[6]))
        x4_SkillList = np.append(x4_SkillList , int(row[7]))

print(x1_AgeList)
print(x2_HeightList)
print(x3_MentalStrengthList)
print(x4_SkillList)

combined_matrix = np.column_stack((np.ones(len(x1_AgeList)), x1_AgeList, x2_HeightList, x3_MentalStrengthList, x4_SkillList))
np.set_printoptions(precision=4)

print(combined_matrix)

# define a function to find residual error and y predictions
def calculator_coef(matrix, out):
    matrix_t = matrix.transpose()

    coef = np.dot(np.dot(np.linalg.inv(np.dot(matrix_t, matrix)), matrix_t), out)
    y_prediction = np.dot(matrix, coef)
    error = np.abs(out - y_prediction)

    plt.scatter(y_prediction, error)
    plt.title("Residual Error Plot")
    plt.xlabel("Predictions")
    plt.ylabel("Error")

    return y_prediction, plt






