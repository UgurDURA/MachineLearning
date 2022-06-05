import pandas as pd
import numpy as np
from sklearn.svm import SVC

data = pd.read_csv('players_svm.csv')
age_x = data.to_numpy()[:, 4]
mental_x = data.to_numpy()[:, 6]
salary_y = data.to_numpy()[:, 9]

combined_matrix = np.column_stack((age_x, mental_x))
matrix = np.array(combined_matrix, dtype=np.int)

train_x = matrix[:80, ]
train_y = salary_y[:80, ]

test_x = matrix[80:, ]
test_y = salary_y[80:, ]

svc_obj = SVC(kernel="linear")
svc_obj.fit(train_x, train_y)

tn = 0
tp = 0
fn = 0
fp = 0

pred_arr = svc_obj.predict(test_x)

for i in range(len(pred_arr)):
    if(pred_arr[i] == 'N' and test_y[i] == 'N'):
        tn = tn + 1
    if(pred_arr[i] == 'N' and test_y[i] == 'Y'):
        fn = fn + 1
    if (pred_arr[i] == 'Y' and test_y[i] == 'Y'):
        tp = tp + 1
    if (pred_arr[i] == 'Y' and test_y[i] == 'N'):
        fp = fp + 1

accuracy = (tn + tp) / (tn + tp + fn + fp)
precision = tp / (tn + tp)
recall = tp / (fn + tp)
specificity = tn / (fp + tn)

F1 = 2 * (recall * precision) / (recall + precision)

print(accuracy)
print(round(precision, 2))
print(round(recall, 2))
print(specificity)
print(round(F1, 2))
