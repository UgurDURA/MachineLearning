def warn(*args, **kwargs):
    pass
from cProfile import label
from unittest import result
import warnings
from sklearn import linear_model

from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
warnings.warn = warn
import csv
from nis import match
from posixpath import split
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
from sklearn.metrics import mean_squared_error
from sklearn.tree import export_text
import rfpimp
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import  LinearRegression, Ridge, RidgeCV, Lasso, LassoCV
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)   
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression



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


# print(X_1)
# print(X_2)
# print(X_3)
# print(X_4)
# print(X_5)
# print(X_6)
# print(Y)

X_combined_matrix = np.column_stack((X_1, X_2, X_3, X_4, X_5, X_6,))                            #[Comment] Combine the numpy arrays into one matrix
print(X_combined_matrix)

X_combined_matrix_with_Y = np.column_stack((X_1, X_2, X_3, X_4, X_5, X_6,Y))      
pandasDF = pd.DataFrame(X_combined_matrix, columns=['X1', 'X2', 'X3', 'X4', 'X5', 'X6'])        #[Comment] Alter the numpy to pandas data frame for further examinations
pandasDF_with_Y = pd.DataFrame(X_combined_matrix_with_Y, columns=['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'Y'])   
print(pandasDF)


############################################################################################################################################
                                            #Statistical Analysis to Define Distribution Type (Q-Q Plots and Wilk Shapiro Test)
############################################################################################################################################
 
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
                                            #Examine the Correlation 
############################################################################################################################################

# correlationResult = pandasDF_with_Y.corr(method='spearman')

# figure = sns.heatmap(correlationResult, cmap = "Blues", annot = True, xticklabels = correlationResult.columns, yticklabels = correlationResult.columns).get_figure()

# figure.savefig("CorrelationMatrix_Sperman5.png", dpi = 1200)

# plt.show

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

############################################################################################################################################
                                            #MTrain Test Split  
############################################################################################################################################

X_train = np.array([])
X_test = np.array([])
Y_train = np.array([]) 
Y_test = np.array([])

def SplitData(X_Matrix, Y, testSize):
    X_train, X_test, Y_train, Y_test= train_test_split(X_Matrix,Y, test_size=testSize, shuffle=False)
    return X_train, X_test, Y_train, Y_test
   

 ############################################################################################################################################
                                            #Selection of Accuracy and Error Metrics
############################################################################################################################################

def calculator_error(y_actual, y_pred, metric):
    rss, tss = 0, 0

    rss = sum((y_actual - y_pred) ** 2)
    tss = sum((y_actual - np.mean(y_actual)) ** 2)

    r_square = 1 - (rss / tss)
    MAE = np.mean(sum(np.abs(y_actual - y_pred)))
    MSE = np.mean(sum((y_actual - y_pred) ** 2))
    RMSE = np.sqrt(MSE)

    if metric == "RSquare":
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
                                            #Cross Validation  
############################################################################################################################################
from sklearn.model_selection import KFold

def get_score(model, X_train, X_test, Y_train, Y_test):
    model.fit(X_train, Y_train)
    return model.score(X_test, Y_test)

 

def Kfold_CrossValidation_Model_R2(model, X_Matrix, Y, cv):
    CV_results = np.array([])
    kf = KFold(n_splits=cv)
    
    for train_index, test_index in kf.split(X_Matrix, Y):
        X_train, X_test, Y_train, Y_test = X_Matrix[train_index], X_Matrix[test_index],Y[train_index], Y[test_index]
        CV_results = np.append(CV_results, get_score(model, X_train, X_test, Y_train, Y_test))
        print(CV_results)
    return CV_results



############################################################################################################################################
                                            #Multiple Linear Regression with Different Train-Test Split 
############################################################################################################################################


def mullin_coef(X, y):
    # Calculating coefficients using the linear algebra equation:
    B_hat = np.dot(X.T, X)
    B_hat = np.linalg.inv(B_hat)
    B_hat = np.dot(B_hat, X.T)
    B_hat = np.dot(B_hat, y)

    return B_hat


coefficients = mullin_coef(X_Matrix, Y)
Y_predictions_MultiLinearRegression = np.dot(X_Matrix, coefficients)

mse = np.array([], dtype=float64)
r_square = np.array([])
mae = np.array([])
rmse = np.array([])
rss = np.array([])

for i in range(1,10):
    train_data, test_data, train_Y, test_Y = SplitData(X_Matrix, Y, 5 * i)
    coefficients = mullin_coef(train_data, train_Y)
    Y_predictions_MultiLinearRegression = np.dot(test_data, coefficients)
    Y_predictions_RSquare = np.dot(X_Matrix,coefficients) #To check the model fit onto the Dataset
    r_error= calculator_error(Y, Y_predictions_RSquare, "RSquare")
    RSS = sum((Y - Y_predictions_RSquare) ** 2)
    MSE = calculator_error(test_Y, Y_predictions_MultiLinearRegression, "MSE")
    MAE = calculator_error(test_Y, Y_predictions_MultiLinearRegression, "MAE")
    RMSE = calculator_error(test_Y, Y_predictions_MultiLinearRegression, "RMSE")
    rmse = np.append(rmse, RMSE)
    mse = np.append(mse, MSE)
    mae = np.append(mae, MAE)
    rss = np.append(rss, RSS)
    r_square = np.append(r_square, r_error)
    print("For k = ", i*5)
    print("R^2 = ", r_error)
    print("MAE = ", MAE)
    print("MSE = ", MSE)
    print("RMSE = ", RMSE)
    print()

range = np.linspace(1,50,num=9)
fig,ax=plt.subplots(figsize=(6,6))
ax.plot(range, rss)
ax.set_xlabel('Test Size (%)',fontsize=20)
ax.set_ylabel('RSS Calculation',fontsize=20)
ax.set_title('Train/Test Split vs RSS ',fontsize=25)
plt.show()

 

        



kFoldValues = [5, 10, 15, 20]
# for i in kFoldValues:
#     CV_results = np.append(CV_results,MultiLinear_Kfold_CrossValidation(X_Matrix, Y, i, 15) ) 
    



for i in kFoldValues:
    Multilinear_CV = Kfold_CrossValidation_Model_R2(LinearRegression(),X_Matrix,Y,10)
    


print(Multilinear_CV)


a = cross_val_score(RandomForestRegressor(),X_Matrix, Y)


print("---------........=========================+>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Cross Validation with A")
print(a)
