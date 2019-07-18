import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import norm


## Gaussians
def gaussian_density(x, mu, sigma):
    return (1/np.sqrt(2*np.pi*np.power(sigma, 2.))) * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))

def gaussian_probability(mean, stdev, x_low, x_high):
    return norm(loc = mean, scale = stdev).cdf(x_high) - norm(loc = mean, scale = stdev).cdf(x_low)

def plot_gaussian(x, mu, sigma):

    # y = gaussian_density(x, mu, sigma)
    y = mlab.normpdf(x,mu,sigma)
    plt.plot(x, y)
    plt.title('Gaussian Probability Density Function')
    plt.xlabel('x variable')
    plt.ylabel('probability density function')
    plt.show()

def plot_gaussian2(x, mu1, sigma1,mu2,sigma2):

    # y = gaussian_density(x, mu, sigma)
    y1 = norm.pdf(x,mu1,sigma1)
    y2 = norm.pdf(x,mu2,sigma2)
    plt.plot(x, y1,'b')
    plt.plot(x, y2,'r--')
    # y3 = y1*y2
    # y3 = ((y1.sum() + y2.sum()) / 2.0) * (y3.astype(float) / y3.astype(float).sum())
    
    # plt.plot(x, y3,'r--')
    plt.title('Gaussian Probability Density Function')
    plt.xlabel('x variable')
    plt.ylabel('probability density function')
    plt.show()


# mu1 = 120
# mu2 = 200
# sigma1 = 40
# sigma2 = 20
# x = np.linspace(1,280,1000)
# plot_gaussian2(x,mu1,sigma1,mu2,sigma2)




