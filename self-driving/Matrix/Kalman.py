import math
import numpy as np
from Kalman.move import move
from Kalman.sense import sense
from gaussian import plot_gaussian2

##先验概率
mu = 0
sig = 1000
(mu0, sig0) = (mu,sig)

##矫正因子
measurements = [5.0,6.0,7.0,9.0,10.0]
measurement_sig = 4.0

##损耗因子
motion = [1.0,1.0,2.0,1.0,1.0]
motion_sig = 2.0

## sense and move
for index in range(0,len(measurements)):
    (mu,sig) = sense(mu,sig, measurements[index],measurement_sig)
    print("sense: ",(mu,sig))
    (mu,sig) = move(mu,sig, motion[index],motion_sig)
    print("move: ", (mu, sig))

(mu1,sig1) = (mu,sig)
x = np.linspace(-50,50,100)
plot_gaussian2(x,mu0,sig0,mu1,sig1)    

    