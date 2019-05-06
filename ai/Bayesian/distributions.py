import numpy as np
import matplotlib.pyplot as plt

#Continuous Uniform Distribution
def probability_uniform(low_range, high_range, minimum, maximum):

    if( isinstance(low_range,str) or 
        isinstance(high_range,str) or 
        isinstance(minimum,str) or 
        isinstance(maximum,str)):
        
        print('Inputs should be numbers not string')
        return None
    
    if (low_range < minimum or low_range > maximum):
        print('Your low range value must be between minimum and maximum')
        return None
        

    if (high_range < minimum or high_range > maximum):
        print('The high range value must be between minimum and maximum')
        return None

    probability = abs(high_range-low_range)/abs(maximum-minimum)
    return probability

## Gaussians
def gaussian_density(x, mu, sigma):
    return (1/np.sqrt(2*np.pi*np.power(sigma, 2.))) * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))


def plot_gaussian(x, mu, sigma):

    y = gaussian_density(x, mu, sigma)

    plt.plot(x, y)
    plt.title('Gaussian Probability Density Function')
    plt.xlabel('x variable')
    plt.ylabel('probability density function')
    plt.show()

#Calculating Area Under the Curve in Python
from scipy.stats import norm

print(gaussian_density(50, 50, 10))
#using scipy
print(norm(loc = 50, scale = 10).pdf(50))

#Calculating Area Under the Curve Solution
def gaussian_probability(mean, stdev, x_low, x_high):
    return norm(loc = mean, scale = stdev).cdf(x_high) - norm(loc = mean, scale = stdev).cdf(x_low)

