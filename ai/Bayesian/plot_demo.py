#Example: ScatterPlot
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

plt.scatter(x, y)
plt.xlabel('x values')
plt.ylabel('y values')
plt.title('X values versus Y values')
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
plt.show()

#Example: Bar Chart
x = ['apples', 'pears', 'bananas', 
     'grapes', 'melons', 'avocados', 'cherries', 'oranges', 'pumpkins',
    'tomatoes']
y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

plt.bar(x, y)
plt.xlabel('x values')
plt.ylabel('y values')
plt.title('X values versus Y values')
plt.xticks(rotation=70)
plt.show()

#Example: Line Chart
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

plt.plot(x, y)
plt.xlabel('x values')
plt.ylabel('y values')
plt.title('X values versus Y values')
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
plt.show()
