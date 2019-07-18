
#dot product of two vectors
def dot_product(veca,vecb):
    if len(a) != len(b):
        print("Error! Vectors must have the same length!")
        return None
    l = len(a)
    sum = 0;
    for i in range(l):
        sum += veca[i] * vecb[i]

    return sum

a = [3,2,4]
b = [2,5,9]
# a*b should be 3*2 + 2*5 + 4*9 = 52
a_dot_b = dot_product(a,b)
print(a_dot_b)

    
    