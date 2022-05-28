import csv

with open('Project/data.csv', encoding="utf8", errors='ignore') as f:
    data = list(csv.reader(f))

print(data)