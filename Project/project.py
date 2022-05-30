import csv
from nis import match
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
import pylab as py
import scipy.stats as stats
from scipy.stats import gaussian_kde
from scipy.stats import shapiro
from numpy import float64, linspace
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_text
import rfpimp


np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   
from statsmodels.stats.outliers_influence import variance_inflation_factor
 



############################################################################################################################################
                                            #Obtain the Data and Assign Accordingly 
############################################################################################################################################

sampleNo = np.array([], dtype=float64)
X_1= np.array([],dtype=float64)
X_2 = np.array([],dtype=float64)
X_3 = np.array([],dtype=float64)
X_4  = np.array([],dtype=float64)
X_5 = np.array([],dtype=float64)
X_6 = np.array([],dtype=float64)
Y= np.array([],dtype=float64)

with open('Project/data.csv', encoding="utf8", errors='ignore') as f:                       #[Comment] Get the data as Numpy 
    data = list(csv.reader(f, delimiter=';'))

print(data)

for row in data:
    if row != data[0]:
        sampleNo= np.append(sampleNo,int(row[0]))                                            #[Comment] Assign each value into different numpy arrays
        X_1= np.append(X_1,int(row[1]))
        X_2= np.append(X_2, int(row[2]))
        X_3= np.append(X_3,int(row[3]))
        X_4= np.append(X_4, int(row[4]))
        X_5= np.append(X_5,int(row[5]))
        X_6= np.append(X_6, int(row[6]))
        Y= np.append(Y, int(row[7]))


print(X_1)
print(X_2)
print(X_3)
print(X_4)
print(X_5)
print(X_6)
print(Y)

X_combined_matrix = np.column_stack((X_1, X_2, X_3, X_4, X_5, X_6,))                            #[Comment] Combine the numpy arrays into one matrix
print(X_combined_matrix)

pandasDF = pd.DataFrame(X_combined_matrix, columns=['X1', 'X2', 'X3', 'X4', 'X5', 'X6'])        #[Comment] Alter the numpy to pandas data frame for further examinations

print(pandasDF)

############################################################################################################################################
                                            #Examine the Histograms and distribution density functions
############################################################################################################################################

# res = pd.Series(pandasDF['X1'], name= "X1")
# plot = sns.displot(data = res)
# plt.show()

# res = pd.Series(pandasDF['X2'], name= "X2")
# plot = sns.displot(data = res )
# plt.show()

# res = pd.Series(pandasDF['X3'], name= "X3")
# plot = sns.displot(data = res )
# plt.show()

# res = pd.Series(pandasDF['X4'], name= "X4")
# plot = sns.displot(data = res)
# plt.show()

# res = pd.Series(pandasDF['X5'], name= "X5")
# plot = sns.displot(data = res)
# plt.show()

# res = pd.Series(pandasDF['X6'], name= "X6")
# plot = sns.displot(data = res)
# plt.show()

# Pandas_Y = pd.DataFrame(Y, columns=['Y'])
# print(Pandas_Y)


# plot = sns.displot(data = Pandas_Y)
# plt.show()

 
 
 

# plt.plot(X_1)
# plt.plot(X_2)
# plt.plot(X_3)
# plt.plot(X_4)
# plt.plot(X_5)
# plt.plot(X_6)
# plt.show()
############################################################################################################################################
                                            #Statistical Analysis to Define Distribution Type (Q-Q Plots and Wilk Shapiro Test)
############################################################################################################################################

 
# stats.probplot(X_1, dist="norm", plot=py)
# py.show()

# stats.probplot(X_2, dist="norm", plot=py)
# py.show()

# stats.probplot(X_3, dist="norm", plot=py)
# py.show()

# stats.probplot(X_4, dist="norm", plot=py)
# py.show()

# stats.probplot(X_5, dist="norm", plot=py)
# py.show()

# stats.probplot(X_6, dist="norm", plot=py)
# py.show()

# stats.probplot(Y, dist="norm", plot=py)
# py.show()
 
 
Shapiro_X1 = shapiro(X_1)
Shapiro_X2 = shapiro(X_2)
Shapiro_X3 = shapiro(X_3)
Shapiro_X4 = shapiro(X_4)
Shapiro_X5 = shapiro(X_5)
Shapiro_X6 = shapiro(X_6)
Shapiro_Y = shapiro(Y)


print("Shapiro Result for X1: ",Shapiro_X1)
print("Shapiro Result for X2: ",Shapiro_X2)
print("Shapiro Result for X3: ",Shapiro_X3)
print("Shapiro Result for X4: ",Shapiro_X4)
print("Shapiro Result for X5: ",Shapiro_X5)
print("Shapiro Result for X6: ",Shapiro_X6)
print("Shapiro Result for Y: ",Shapiro_Y)


############################################################################################################################################
                                            #Plotting Probability Density Function (PDF) onto Histogram Distributions
############################################################################################################################################


def calculate_PDF_Parametric(Column):
    mu, std = norm.fit(Column) 
    # Plot the histogram.
    plt.hist(X_1, bins=25, density=True, alpha=0.6, color='b')
  
    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
  
    plt.plot(x, p, 'k', linewidth=2)
    title = "Fit Values: {:.2f} and {:.2f}".format(mu, std)
    plt.title(title)
  
    plt.show()

def calculate_PDF_NonParametric(columnName):
     
    res = pd.Series(pandasDF[columnName], name= columnName)
    plot = sns.displot(data = res, kde = True, color='b')
    plt.show()


# calculate_PDF_NonParametric("X1")
# calculate_PDF_NonParametric("X2")
# calculate_PDF_NonParametric("X3")
# calculate_PDF_NonParametric("X4")
# calculate_PDF_Parametric(X_5)
# calculate_PDF_NonParametric("X6")


 
# sns.set(style='ticks')

# # parameterise our distributions
# d1 = stats.norm(Y)
 

# # sample values from above distributions
# y1 = d1.rvs(100)
 
# # create new figure with size given explicitly
# plt.figure(figsize=(10, 6))

# # add histogram showing individual components
# plt.hist([y1], 31, histtype='barstacked', density=True, alpha=0.4, edgecolor='none')

# # get X limits and fix them
# mn, mx = plt.xlim()
# plt.xlim(mn, mx)

# # add our distributions to figure
# x = np.linspace(mn, mx, 100)
# plt.plot(x, d1.pdf(x) , color='C0', ls='--', label='d1')
 

# # estimate Kernel Density and plot
# kde = stats.gaussian_kde(Y)
# plt.plot(x, kde.pdf(x), label='KDE')

# # finish up
# plt.legend()
# plt.ylabel('Probability density')
# sns.despine()

# plt.show()
 
 
 







 
############################################################################################################################################
                                            #Examine the Correlation 
############################################################################################################################################

# correlationResult = pandasDF.corr(method='spearman')

# figure = sns.heatmap(correlationResult, cmap = "Blues", annot = True, xticklabels = correlationResult.columns, yticklabels = correlationResult.columns).get_figure()

# figure.savefig("CorrelationMatrix_Sperman.png", dpi = 1200)

vif_data = pd.DataFrame()
vif_data["feature"] = pandasDF.columns
vif_data["VIF"] = [variance_inflation_factor(pandasDF.values, i)
                          for i in range(len(pandasDF.columns))]
  
print(vif_data)


 
############################################################################################################################################
                                            #Discard the Correlated Column
############################################################################################################################################


pandasDF.__delitem__("X6")
X_Matrix = np.delete(X_combined_matrix, 5, 1)

print(pandasDF)
print(X_Matrix)

############################################################################################################################################
                                            #Selection of Accuracy and Error Metrics
############################################################################################################################################

def calculator_error(y_actual, y_pred, metric):
    rss, tss = 0, 0

    average_y = np.average(y_actual)

    rss = sum((y_actual - y_pred) ** 2)
    tss = sum((y_actual - np.mean(y_actual)) ** 2)

    r_square = 1 - (rss / tss)
    MAE = np.mean(sum(np.abs(y_actual - y_pred)))
    MSE = np.mean(rss)
    RMSE = np.sqrt(MSE)

    if metric == "rSquare":
        return r_square
    elif metric == "MSE":
        return MSE
    elif metric == "MAE":
        return MAE
    elif metric == "RMSE":
        return RMSE
    else:
        return 0


############################################################################################################################################
                                            #MTrain Test Split with k-Fold Validation
############################################################################################################################################



def kFold(matrix, y, k):

    splited_matrix = np.array_split(matrix, k)
    train_data = np.concatenate(np.delete(splited_matrix, k-1, axis=0))
    test_data = splited_matrix[k-1]
    y_splitted = np.array_split(y,k)
    train_Y = np.concatenate(np.delete(y_splitted, k-1, axis=0))
    test_Y = y_splitted[k-1]

    return train_data, test_data, train_Y, test_Y

# train_data, test_data, train_Y, test_Y = kFold(X_Matrix,Y,60)

# print("Training Data")
# print(train_data)
# print("\n Test Data")
# print(test_data)

############################################################################################################################################
                                            #Multiple Linear Regression
############################################################################################################################################


def MultipleLinearRegression(X_matrix, Y_actual):
    B_hat = np.linalg.inv(np.dot(X_matrix.T, X_matrix))
    B_hat = np.dot(B_hat, X_matrix.T)
    B_hat = np.dot(B_hat, Y_actual)

    return B_hat

train_data, test_data, train_Y, test_Y = kFold(X_Matrix, Y, 25)
coefficients = MultipleLinearRegression(train_data, train_Y)
Y_predictions = np.dot(train_data, coefficients)
print("Y Predictions : ")
print(Y_predictions)


# r_error= calculator_error(train_Y, Y_predictions, "rSquare")
# MSE = calculator_error(train_Y, Y_predictions, "MSE")
# MAE = calculator_error(train_Y, Y_predictions, "MAE")
# RMSE = calculator_error(train_Y, Y_predictions, "RMSE")


# print("R^2 ==> ", r_error)
# print("MAE = ", MAE)
# print("MSE = ", MSE)
# print("RMSE = ", RMSE)


############################################################################################################################################
                                            #Multiple Linear Regression with Only Normal Distributions (Predictors)
############################################################################################################################################

# X_Matrix = np.delete(X_Matrix, 4, 1)

# print(X_Matrix)

# train_data, test_data, train_Y, test_Y = kFold(X_Matrix,Y,10)

# coefficients = MultipleLinearRegression(train_data, train_Y)
# Y_predictions = np.dot(train_data, coefficients)
# print("Y Predictions : ")
# print(Y_predictions)

# r_error= calculator_error(train_Y, Y_predictions, "rSquare")
# MSE = calculator_error(train_Y, Y_predictions, "MSE")
# MAE = calculator_error(train_Y, Y_predictions, "MAE")
# RMSE = calculator_error(train_Y, Y_predictions, "RMSE")


# print("R^2 ==> ", r_error)
# print("MAE = ", MAE)
# print("MSE = ", MSE)
# print("RMSE = ", RMSE)

############################################################################################################################################
                                            #Multiple Linear Regression with Cross Validation and K-Fold
############################################################################################################################################
# from 5 to 20
mse = np.array([], dtype=float64)
r_square = np.array([])
mae = np.array([])
rmse = np.array([])

print("Multiple Regression with k-fold Validation")
for i in range(1, 5):
    train_data, test_data, train_Y, test_Y = kFold(X_Matrix, Y, 5 * i)
    coefficients = MultipleLinearRegression(train_data, train_Y)
    Y_predictions = np.dot(train_data, coefficients)
    r_error= calculator_error(train_Y, Y_predictions, "rSquare")
    MSE = calculator_error(train_Y, Y_predictions, "MSE")
    MAE = calculator_error(train_Y, Y_predictions, "MAE")
    RMSE = calculator_error(train_Y, Y_predictions, "RMSE")
    rmse = np.append(rmse, RMSE)
    mse = np.append(mse, MSE)
    mae = np.append(mae, MAE)
    r_square = np.append(r_square, r_error)
    print("For k = ", i*5)
    print("R^2 = ", r_error)
    print("MAE = ", MAE)
    print("MSE = ", MSE)
    print("RMSE = ", RMSE)
    print()


 
 
# plt.plot(r_square)
# plt.show()

# plt.plot(mse)
# plt.show()

# plt.plot(mae)
# plt.show()

# plt.plot(rmse)
# plt.show()



############################################################################################################################################
                                            #Random Forest 
############################################################################################################################################


def MSE(y, y_prediction):
    total = 0
    for i in range(len(y)):
       total += (y[i]-y_prediction[i]) ** 2
    mse = total / len(y)
    return mse

 
train_data, test_data, train_Y, test_Y = kFold(X_Matrix, Y, 10)


randomForestRegressor = RandomForestRegressor(max_depth=12, random_state=0)
randomForestRegressor.fit(train_data,train_Y)

pred_Y = randomForestRegressor.predict(test_data)


r_error= calculator_error(test_Y, pred_Y, "rSquare")

print("==========================> R^2 for Random Forest = ", r_error)

print(pred_Y)


MSE_Result = MSE(test_Y,pred_Y )

MSE_Results = []

MSE_Results = np.append(MSE_Results, MSE_Result)

 
for i in range(1,15):
    reg = RandomForestRegressor( max_depth=i, random_state=0)
    reg.fit(train_data, train_Y)
    pred_Y = reg.predict(test_data)
    MSE_result = MSE(test_Y, pred_Y)
    MSE_Results = np.append(MSE_Results, MSE_result)
    print("MSE with depth ",i, ":    ",MSE_Results[i])

min_value = min(MSE_Results)
if min_value in MSE_Results:
    value_index =np.where(MSE_Results == min_value)
print("Optimum depth value : ", value_index[0])


randomForestRegressor_1 = RandomForestRegressor(n_estimators = 1,max_depth=1, random_state=0)
randomForestRegressor_1.fit(train_data,train_Y)
randomForestRegressor_2 = RandomForestRegressor(n_estimators = 1,max_depth=2, random_state=0)
randomForestRegressor_2.fit(train_data,train_Y)
randomForestRegressor_3 = RandomForestRegressor(n_estimators = 1,max_depth=6, random_state=0)
randomForestRegressor_3.fit(train_data,train_Y)
randomForestRegressor_4 = RandomForestRegressor(n_estimators = 1,max_depth=8, random_state=0)
randomForestRegressor_4.fit(train_data,train_Y)
randomForestRegressor_5 = RandomForestRegressor(n_estimators = 1,max_depth=10, random_state=0)
randomForestRegressor_5.fit(train_data,train_Y)
randomForestRegressor_6 = RandomForestRegressor(n_estimators = 1,max_depth=12, random_state=0)
randomForestRegressor_6.fit(train_data,train_Y)

 

underlying_tree1 = randomForestRegressor_1.estimators_
underlying_tree2 = randomForestRegressor_2.estimators_
underlying_tree3 = randomForestRegressor_3.estimators_
underlying_tree4 = randomForestRegressor_4.estimators_
underlying_tree5 = randomForestRegressor_5.estimators_
underlying_tree6 = randomForestRegressor_6.estimators_
features_X = ["X1", "X2"," X3", 'X4','X5']
tree1 = export_text(underlying_tree1[0],feature_names= features_X)
tree2 = export_text(underlying_tree2[0],feature_names= features_X)
tree3 = export_text(underlying_tree3[0],feature_names= features_X)
tree4 = export_text(underlying_tree4[0],feature_names= features_X)
tree5 = export_text(underlying_tree5[0],feature_names= features_X)
tree6 = export_text(underlying_tree6[0],feature_names= features_X)

print("tree with depth 1:")
print(tree1)
print("tree with depth 3:")
print(tree2)
print("tree with depth 4:")
print(tree3)
print("tree with depth 6:")
print(tree4)

print("tree with depth 8:")
print(tree5)
print("tree with depth 12:")
print(tree6)


plt.plot(MSE_Results)
plt.show()


############################################################################################################################################
                                            #Random Forest with K-fold Cross Validation
############################################################################################################################################

mse = np.array([], dtype=float64)
r_square = np.array([])
mae = np.array([])
rmse = np.array([])



print("Random Forest with k-fold Validation")
for i in range(1, 11):
    train_data, test_data, train_Y, test_Y = kFold(X_Matrix, Y, 5 * i)
    randomForestRegressor = RandomForestRegressor(max_depth=12, random_state=0)
    randomForestRegressor.fit(train_data,train_Y)
    Y_predictions = randomForestRegressor.predict(test_data)
    r_error= calculator_error(test_Y, Y_predictions, "rSquare")
    MSE = calculator_error(test_Y, Y_predictions, "MSE")
    MAE = calculator_error(test_Y, Y_predictions, "MAE")
    RMSE = calculator_error(test_Y, Y_predictions, "RMSE")
    rmse = np.append(rmse, RMSE)
    mse = np.append(mse, MSE)
    mae = np.append(mae, MAE)
    r_square = np.append(r_square, r_error)
    print("For k = ", i*5)
    print("R^2 = ", r_error)
    print("MAE = ", MAE)
    print("MSE = ", MSE)
    print("RMSE = ", RMSE)
    print()

plt.plot(r_square)
plt.show()

plt.plot(mse)
plt.show()

plt.plot(mae)
plt.show()

plt.plot(rmse)
plt.show()