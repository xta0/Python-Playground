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

# R = np.random.random(10000)
# plt.hist(R)
# plt.show()



t_y = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_x = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
plt.scatter(t_x, t_y)
plt.show()
