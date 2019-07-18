import random as rd
import matplotlib.pyplot as plt

def simulate_dice_rolls(N):
    roll_counts = [0,0,0,0,0,0]
    for i in range(N):
        roll = rd.choice([1,2,3,4,5,6])
        index = roll - 1
        roll_counts[index] += 1;
    return roll_counts

def show_roll_data(roll_counts):
    number_of_sides_on_die = len(roll_counts)
    for i in range(number_of_sides_on_die):
        number_of_rolls = roll_counts[i]
        number_on_die = i+1
        print(number_on_die, "came up", number_of_rolls, "times")
        
roll_data = simulate_dice_rolls(10)
print(roll_data)
show_roll_data(roll_data)


# %matplotlib inline

def visualize_one_die(roll_data):
    roll_outcomes = [1,2,3,4,5,6]
    fig, ax = plt.subplots()
    ax.bar(roll_outcomes, roll_data)
    ax.set_xlabel("Value on Die")
    ax.set_ylabel("# rolls")
    ax.set_title("Simulated Counts of Rolls")
    plt.show()
    
roll_data = simulate_dice_rolls(500)
visualize_one_die(roll_data)

p=[0.2, 0.2, 0.2, 0.2, 0.2]
world=['green', 'red', 'red', 'green', 'green']
Z = 'red'
pHit = 0.6
pMiss = 0.2

