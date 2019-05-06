import random

def binary_search(array, num):
    first = 0
    last = len(array)-1
    
    while first <= last:
        mid = int((first+last)/2)
        if array[mid] == num:
            return mid
        elif array[mid] > num:
            last = mid - 1
        else:
            first = mid + 1

    return -1

def binary_search_recursive(array,num,lo,hi):
    if lo > hi:
        return -1
    mid = int((lo+hi)/2)
    if array[mid] == num :
        return mid
    elif array[mid] > num:
        return binary_search_recursive(array,num,lo,mid-1)
    else:
        return binary_search_recursive(array,num,mid+1,hi)

array = random.sample(range(1,100),10)
print(array)
array.sort()
print(array)
print(binary_search(array,array[3]) == 3) 
print(binary_search_recursive(array,array[3],0,9) == 3) 