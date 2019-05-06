import random 

def mergeSort(array):
    if len(array) <= 1:
        return
    left = array[:int(len(array)/2)]
    right = array[int(len(array)/2):]

    mergeSort(left)
    mergeSort(right)

    li = ri = i = 0
    while li < len(left) and ri <len(right):
        if left[li] <= right[ri]:
            array[i] = left[li]
            li += 1
        else:
            array[i] = right[ri]
            ri += 1
        i += 1

    #append left
    while( li < len(left) ):
        array[i] = left[li]
        li+=1
        i+=1
    
    #append right
    while( ri < len(right) ):
        array[i] = right[ri]
        ri+=1
        i+=1

array = random.sample(range(1,100),20)
print(array)
mergeSort(array)
print(array)

