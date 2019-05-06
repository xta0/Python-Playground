import os

a = 10
b = 10
print(hex(id(a)))
print(hex(id(b)))
print(type(a))
a = 15
print(hex(id(a)))

my_list = [1,2,3]
print(hex(id(my_list)))
my_list.append(4)
print(hex(id(my_list))) #
#2
my_list = my_list+[4]
print(hex(id(my_list)))

my_str = "123"
print(hex(id(my_str)))
my_str += "4"
print(my_str)
print(hex(id(my_str)))
