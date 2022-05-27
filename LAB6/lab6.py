import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor

salary= np.array([])
age = np.array([])
height = np.array([])
mental  = np.array([])
skill = np.array([])


def MSE(y, y_prediction):
    total = 0
    for i in range(len(y)):
       total += (y[i]-y_prediction[i]) ** 2
    mse = total / len(y)
    return mse


with open('LAB3/Football_players(4).csv', encoding="utf8", errors='ignore') as f:
    data = list(csv.reader(f))

print(data)

for row in data:
    salary = np.append(salary, int(row[8]))
    age= np.append(age, int(row[4]))
    height = np.append(height, int(row[5]))
    mental = np.append(mental, int(row[6]))
    skill = np.append(skill, int(row[7]))

X_combined_matrix = np.column_stack((age,height, mental, skill))

train_X = X_combined_matrix[:80, ]
train_Y = salary[:80, ]

test_X = X_combined_matrix[80:,]
test_Y = salary[80:,]


randomForestRegressor = RandomForestRegressor(n_estimators=1, max_depth=3, random_state=0)

array = [1, 10, 100, 200, 400, 700, 1000, 5000]
MSE_Results = []
for i in range(len(array)):
    k = array[i]
    reg = RandomForestRegressor(n_estimators=k, max_depth=3, random_state=0)
    reg.fit(train_X, train_Y)
    y_pred = reg.predict(test_X)
    mse_value = MSE(test_Y, y_pred)
    MSE_Results = np.append(MSE_Results, mse_value)
    print(reg.feature_importances_)
    print("MSE with: ")
    print(k)
    print(y_pred)
    print(mse_value)


for i in range(len(MSE_Results)):
    print(MSE_Results[i])

plt.figure()
plt.plot([1, 2, 3, 4, 5, 6, 7, 8], MSE_Results)
plt.show()





 


