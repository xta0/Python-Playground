# Step 1.
#  we import pyplot as plt so we can refer to the pyplot
#  succinctly. This is a standard convention for this library.

from matplotlib import pyplot as plt

X = [
    2.000,
    2.333,
    2.667,
    3.000
]

Y = [
    30,
    40,
    68,
    80
]

plt.scatter(X,Y)
plt.plot(X,Y)
plt.title("Position vs. Time on a Roadtrip")
plt.xlabel("Time (in hours)")
plt.ylabel("Odometer Reading (in miles)")
plt.show()