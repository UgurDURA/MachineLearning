import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
import pylab as py
import scipy.stats as stats
from scipy.stats import gaussian_kde
from scipy.stats import shapiro
 



############################################################################################################################################
                                            #Obtain the Data and Assign Accordingly 
############################################################################################################################################

sampleNo = np.array([])
X_1= np.array([])
X_2 = np.array([])
X_3 = np.array([])
X_4  = np.array([])
X_5 = np.array([])
X_6 = np.array([])
Y= np.array([])

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
 
Shapiro_X1 = shapiro(X_1)
Shapiro_X2 = shapiro(X_2)
Shapiro_X3 = shapiro(X_3)
Shapiro_X4 = shapiro(X_4)
Shapiro_X5 = shapiro(X_5)
Shapiro_X6 = shapiro(X_6)

print("Shapiro Result for X1: ",Shapiro_X1)
print("Shapiro Result for X2: ",Shapiro_X2)
print("Shapiro Result for X3: ",Shapiro_X3)
print("Shapiro Result for X4: ",Shapiro_X4)
print("Shapiro Result for X5: ",Shapiro_X5)
print("Shapiro Result for X6: ",Shapiro_X6)

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


calculate_PDF_NonParametric("X1")
calculate_PDF_NonParametric("X2")
calculate_PDF_NonParametric("X3")
calculate_PDF_NonParametric("X4")
calculate_PDF_Parametric(X_5)
calculate_PDF_NonParametric("X6")







 
############################################################################################################################################
                                            #Examine the Correlation 
############################################################################################################################################

correlationResult = pandasDF.corr(method='spearman')

figure = sns.heatmap(correlationResult, cmap = "Blues", annot = True, xticklabels = correlationResult.columns, yticklabels = correlationResult.columns).get_figure()

# figure.savefig("CorrelationMatrix_Sperman.png", dpi = 1200)


 
############################################################################################################################################
                                            #Discard the Correlated Column
############################################################################################################################################


pandasDF.__delitem__("X6")

print(pandasDF)