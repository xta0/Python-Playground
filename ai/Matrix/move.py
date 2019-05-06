import numpy as np
from car.Car import Car

# Create a 2D world of 0's
height = 4
width = 6
world = np.zeros((height, width))

# Define the initial car state
initial_position = [0, 0] # [y, x] (top-left corner)
velocity = [0, 1] # [vy, vx] (moving to the right)


cara =  Car(initial_position,velocity,world)
cara.move()
cara.move()
cara.turn_right()
cara.move()
cara.move()
cara.move()
cara.turn_right()
cara.move()
cara.move()
cara.turn_right()
cara.move()
cara.move()
cara.move()
cara.display_world()