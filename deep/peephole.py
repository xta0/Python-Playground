import string
import time
letters = string.ascii_letters
char_list = list(letters)
char_tuple = tuple(letters)
char_set = set(letters)
print(char_list)

def membership_test(n,container):
    for i in range(n):
        if 'z' in container:
            pass

#test array
start = time.perf_counter()
membership_test(1000000,char_list) #一百万次
end = time.perf_counter()
print('list:',end-start)

#test tuple
start = time.perf_counter()
membership_test(1000000,char_tuple) 
end = time.perf_counter()

#test set
print('set:',end-start)
start = time.perf_counter()
membership_test(1000000,char_set)
end = time.perf_counter()
print('list:',end-start)
