import matplotlib.pyplot as plt

def draw(p):
    x = ['1','2','3','4','5']
    y = p

    plt.bar(x, y)
    plt.xlabel('x values')
    plt.ylabel('y values')
    plt.ylim(0.0,1.0)
    plt.title('probability distribution')
    plt.show()