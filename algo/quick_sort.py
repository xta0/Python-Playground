import random 

def quick_sort(array, left, right):
    if left >= right:
        return 
    #choose the left most number as a pivot
    pivot = array[left]
    l = left 
    r = right    
    while l<r:
        #move right part first
        while array[r] >= pivot and r>l:
            r -=1
        #then move left part
        while array[l] <= pivot and l<r:
            l+=1
        if r>l:
            #swap
            array[l],array[r] = array[r], array[l]

    array[left],array[r] = array[r],array[left]

    #recursive 
    quick_sort(array,left,l-1)
    quick_sort(array,l+1, right)


array = random.sample(range(0,100),10)
print(array)
quick_sort(array,0,len(array)-1)
print(array)