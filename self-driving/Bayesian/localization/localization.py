#Modify the previous code so that the robot senses red twice.
from move import move
from sense import sense
from draw import draw


p=[0.2, 0.2, 0.2, 0.2, 0.2]

world=['green', 'red', 'red', 'green', 'green']

measurements = ['red','green']
motions = [1,1]

# for k in range(len(measurements)):
#     #sense
#     p = sense(p,measurements[k])
#     p = move(p,motions[k])
p = sense(p,'red')
# p = move(p,1)
# p = sense(p,'green')



draw(p)