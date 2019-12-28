import numpy as np 

# data = []
# for line in open("data_2d.csv"):
#     row = line.rstrip().split(',')
#     sample = list(map(float, row))
#     data.append(sample)
# # turns into numpy array
# data = np.array(data)
# print(data)

import pandas as pd

data = pd.read_csv("data_2d.csv", header=None)
# print(data.info())
# print(data.head())
# print(type(data[[0,2]]))
# print(data[data[0]<5])
# M = data.as_matrix()
# print(type(M))
# data[3] = data.apply(lambda col: col[0]*col[1], axis=1)
# print(data.head())

t1 = pd.read_csv("table1.csv")
t2 = pd.read_csv("table2.csv")
print(t1.info())
print(t2.info())
print(t1.head())
print(t2.head())
m = pd.merge(t1,t2,on="user_id")
print(m.head())

np_arr = data.values()