
import numpy as np

# QUESTION1
# -------------------------------------------------------
arr = np.array([])
arr = np.random.randint(-100,100,25)
     
print(arr)

evens = 0
odds = 0

for elem in arr:
    if elem % 2 == 0:
        evens +=1
    else:
        odds +=1


print("Evens: ",evens)
print("Odds: "+str(odds))

# -------------------------------------------------------
# QUESTION2


# sum_arr = 0

# for i in range(len(arr)):
#      sum_arr += arr[i]

# avg_arr = summ_arr/len(arr)

sum_arr = np.sum(arr)
avg_arr = np.mean(arr)


print("Sum: ",sum_arr)
print("Average:", avg_arr)

