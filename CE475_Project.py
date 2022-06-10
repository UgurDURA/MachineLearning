def warn(*args, **kwargs):
    pass
from unittest import result
import warnings
from sklearn import linear_model

from sklearn.preprocessing import PolynomialFeatures
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
print(X_Matrix)