
import matplotlib.pyplot as plt

# p=[0.2, 0.2, 0.2, 0.2, 0.2]
world=['green', 'red', 'red', 'green', 'green']
pHit = 0.6
pMiss = 0.2

def sense(p, Z):
    q=[]
    for i in range(len(p)):
        hit = (Z == world[i])
        q.append(p[i] * (hit * pHit + (1-hit) * pMiss))
    s = sum(q)
    for i in range(len(q)):
        q[i] = q[i] / s
    return q


# def draw(p):
#     x = ['1','2','3','4','5']
#     y = p
#     plt.bar(x, y)
#     plt.xlabel('x values')
#     plt.ylabel('y values')
#     plt.ylim(0.00,1)
#     plt.title('probability distribution')
#     # plt.xticks(rotation=70)
#     plt.show()

# draw(p)
