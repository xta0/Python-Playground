import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 

# x = np.linspace(0,10,100) #from 0 - 10, 100 points
# y = np.sin(x)
# plt.plot(x,y)
# plt.xlabel("Time")
# plt.ylabel("Some function of Time")
# plt.title("My Chart")
# plt.show()

# A = pd.read_csv('data_1d.csv', header = None).as_matrix()
# x = A[:,0] #1st column
# y = A[:,1] #2nd column
# plt.scatter(x,y)
# plt.show() 

R = np.random.random(10000)
plt.hist(R)
plt.show()