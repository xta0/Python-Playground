import pdb

k = 3
a = [0,1,2,3,4,5,6,7,8,9]

for x in range(0,k):
    tmp = a[9]
    for i,n in enumerate(a):
        index = 9-i
        if index>=1:
            a[index] = a[index-1]
    a[0]=tmp

print(*a, sep = ", ") 