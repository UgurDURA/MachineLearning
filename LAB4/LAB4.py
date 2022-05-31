import numpy as np
import csv
import matplotlib.pyplot as plt


def mullin_coef(X, y):
    # Calculating coefficients using the linear algebra equation:
    B_hat = np.linalg.inv(np.dot(X.T, X))
    B_hat = np.dot(B_hat, X.T)
    B_hat = np.dot(B_hat, y)

    return B_hat


def k_fold_cv(X, y, k):

    cv_hat = np.array([])

    fold_size = int(len(age_list) / k)
    
    for i in range(0, len(age_list), fold_size):  # For each fold:

        X_test = X[i:i + fold_size]  # Determine test input data

        y_test = y[i:i + fold_size]  # Determine test output data
        X_train = np.delete(X, range(i, i + fold_size), 0)  # Determine train input data
        y_train = np.delete(y, range(i, i + fold_size), 0)  # Determine train output data

        # Calculate coefficients using the linear algebra equation (with train input and output)
        B_hat = mullin_coef(X_train, y_train)

        # Calculate predictions with test input
        y_hat = np.dot(X_test, B_hat)

        # Append the predictions to cv_hat
        cv_hat = np.append(cv_hat, y_hat)

    return cv_hat


with open("LAB3/Football_players(4).csv", encoding="utf8", errors='ignore') as f:
    csv_list = list(csv.reader(f))

age_list = np.array([])
hgt_list = np.array([])
mnt_list = np.array([])
skl_list = np.array([])
sal_list = np.array([])

# Extracting data into lists, creating X and y:
for row in csv_list[1:]:
    age_list = np.append(age_list, int(row[4]))
    hgt_list = np.append(hgt_list, int(row[5]))
    mnt_list = np.append(mnt_list, int(row[6]))
    skl_list = np.append(skl_list, int(row[7]))
    sal_list = np.append(sal_list, int(row[8]))

# Forming the input(X) and the output(y)
ones = np.ones((len(age_list)))
X = np.column_stack((ones, age_list, hgt_list, mnt_list, skl_list))
y = sal_list

print()
print(X)
print(y)

p1 = k_fold_cv(X, y, 5)
p2 = k_fold_cv(X, y, 10)

coefficients = mullin_coef(X, y)
p3 = np.dot(X, coefficients)

e1 = y - p1
e2 = y - p2
e3 = y - p3

# ---------------------------------------------------------------------------------
# Plotting

plt.scatter(np.linspace(1, len(e1), len(e1)), e1, c='b', label="Errors w/ 5-fold CV")
plt.scatter(np.linspace(1, len(e2), len(e2)), e2, c='r', label="Errors w/ 10-fold CV")
plt.scatter(np.linspace(1, len(e3), len(e3)), e3, c='g', label="Training errors")
plt.hlines(0, xmin=0, xmax=len(e1), colors='k', label="Zero error line")
plt.title("Plot: Error Values")
plt.xlabel("Prediction no.")
plt.ylabel("Error")
plt.xticks(np.arange(1, len(e1), 2))
plt.legend()
plt.show()