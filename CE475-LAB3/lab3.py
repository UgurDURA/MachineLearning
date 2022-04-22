import csv


with open('CE475-LAB3/Football_players(6).csv', encoding="utf8", errors='ignore') as f:
    data = list(csv.reader(f))

print(data)

