import csv
from inspect import trace
from pyexpat import features
from tkinter.ttk import Treeview
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_text


salary_y= np.array([])
age_x1 = np.array([])
height_x2 = np.array([])
mental_x3  = np.array([])
skill_x4 = np.array([])


def MSE(y, y_prediction):
    total = 0
    for i in range(len(y)):
       total += (y[i]-y_prediction[i]) ** 2
    mse = total / len(y)
    return mse


with open('Football_players(24).csv', encoding="utf8", errors='ignore') as f:
    data = list(csv.reader(f))

print(data)

for row in data:
     if row != data[0]:
        salary_y = np.append(salary_y, int(row[8]))
        age_x1= np.append(age_x1, int(row[4]))
        height_x2 = np.append(height_x2,int(row[5]))
        mental_x3 = np.append(mental_x3, int(row[6]))
        skill_x4 = np.append(skill_x4, int(row[7]))

X_combined_matrix = np.column_stack((age_x1,height_x2, mental_x3, skill_x4))

train_X = X_combined_matrix[:80, ]
train_Y = salary_y[:80, ]

test_X = X_combined_matrix[80:,]
test_Y = salary_y[80:,]

print(X_combined_matrix)
print(train_X)
print(train_Y)
print(test_X)
print(test_Y)

randomForestRegressor = RandomForestRegressor(max_depth=1, random_state=0)
randomForestRegressor.fit(train_X,train_Y)

pred_Y = randomForestRegressor.predict(test_X)

print(pred_Y)

MSE_Result = MSE(test_Y,pred_Y )

MSE_Results = []

MSE_Results = np.append(MSE_Results, MSE_Result)

 
for i in range(1,8):
    reg = RandomForestRegressor( max_depth=i, random_state=0)
    reg.fit(train_X, train_Y)
    pred_Y = reg.predict(test_X)
    MSE_result = MSE(test_Y, pred_Y)
    MSE_Results = np.append(MSE_Results, MSE_result)
    print("MSE with depth ",i, ":    ",MSE_Results[i])

min_value = min(MSE_Results)
if min_value in MSE_Results:
    value_index =np.where(MSE_Results == min_value)
print("Optimum depth value : ", value_index[0])

randomForestRegressor_1 = RandomForestRegressor(n_estimators = 1,max_depth=1, random_state=0)
randomForestRegressor_1.fit(train_X,train_Y)
randomForestRegressor_2 = RandomForestRegressor(n_estimators = 1,max_depth=3, random_state=0)
randomForestRegressor_2.fit(train_X,train_Y)

underlying_tree1 = randomForestRegressor_1.estimators_
underlying_tree2 = randomForestRegressor_2.estimators_
features_X = ["age", "height"," mental", 'skill']
tree1 = export_text(underlying_tree1[0],feature_names= features_X)
tree2 = export_text(underlying_tree2[0],feature_names= features_X)

print("tree with depth 1:")
print(tree1)
print("tree with depth 3:")
print(tree2)

