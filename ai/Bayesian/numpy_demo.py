import math
import numpy as np

### Square 

print('\nExample of taking the square')
print(math.pow(2,2))
print(np.square(2))

print('\nExample of taking the cube')
print(math.pow(2,3))
print(np.power(2,3))

print('\nExample of taking the square root')
print(math.sqrt(4))
print(np.sqrt(4))

print('\nExample of taking the exponent')
print(math.exp(3))
print(np.exp(3))

# Using numpy with lists

print('\nExample of squaring elements in a list')
print(np.square([1, 2, 3, 4, 5]))

print('\nExample of taking the square root of a list')
print(np.sqrt([1, 4, 9, 16, 25]))

print('\nExamples of taking the cube of a list')
print(np.power([1, 2, 3, 4, 5], 3))

# Using numpy in a function

def numpy_example(x):
    return np.exp(x)

x = [1, 2, 3, 4, 5]
print(numpy_example(x))