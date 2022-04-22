import csv
from re import X
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





