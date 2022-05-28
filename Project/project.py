import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm

sampleNo = np.array([])
X_1= np.array([])
X_2 = np.array([])
X_3 = np.array([])
X_4  = np.array([])
X_5 = np.array([])
X_6 = np.array([])
Y= np.array([])

with open('Project/data.csv', encoding="utf8", errors='ignore') as f:
    data = list(csv.reader(f, delimiter=';'))

print(data)

for row in data:
    if row != data[0]:
        sampleNo= np.append(sampleNo,int(row[0]))
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

X_combined_matrix = np.column_stack((X_1, X_2, X_3, X_4, X_5, X_6,))

print(X_combined_matrix)

pandasDF = pd.DataFrame(X_combined_matrix, columns=['X1', 'X2', 'X3', 'X4', 'X5', 'X6'])

print(pandasDF)

res = pd.Series(pandasDF['X1'], name= "X1")
plot = sns.displot(data = res, kde = True)
plt.show()

res = pd.Series(pandasDF['X2'], name= "X2")
plot = sns.displot(data = res, kde = True)
plt.show()

res = pd.Series(pandasDF['X3'], name= "X3")
plot = sns.displot(data = res, kde = True)
plt.show()

res = pd.Series(pandasDF['X4'], name= "X4")
plot = sns.displot(data = res, kde = True)
plt.show()

res = pd.Series(pandasDF['X5'], name= "X5")
plot = sns.displot(data = res, kde = True)
plt.show()

res = pd.Series(pandasDF['X6'], name= "X6")
plot = sns.displot(data = res, kde = True)
plt.show()
 
 

plt.plot(X_1)
# plt.plot(X_2)
# plt.plot(X_3)
# plt.plot(X_4)
# plt.plot(X_5)
plt.plot(X_6)
plt.show()




correlationResult = pandasDF.corr()

figure = sns.heatmap(correlationResult, cmap = "Blues", annot = True, xticklabels = correlationResult.columns, yticklabels = correlationResult.columns).get_figure()

# figure.savefig("CorrelationMatrix.png", dpi = 1200)


