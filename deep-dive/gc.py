import ctypes
import gc

def ref_count(address):
    return ctypes.c_long.from_address(address).value

def object_by_id(object_id):
    for  obj in gc.get_objects():
        if id(obj) == object_id:
                return "Object Exists"
    return "Not Found"

class A:
    def __init__(self):
        self.b = B(self)
        print(f'A: self:{hex(id(self))},b:{hex(id(self.b))}')

class B:
    def __init__(self, a):
        self.a = a
        print(f'B: self:{hex(id(self))},a:{hex(id(self.a))}')

gc.disable()
a  = A()

id_a = id(a)
id_b = id(a.b)
print(hex(id_a))
print(hex(id_b))

print(ref_count(id_a))
print(ref_count(id_b))
print(object_by_id(id_a))
print(object_by_id(id_b))

a = None
print(ref_count(id_a))
print(ref_count(id_b))

gc.collect()
print(object_by_id(id_a))
print(object_by_id(id_b))


print(hex(id_a))
print(hex(id_b))
print(ref_count(id_a))
print(ref_count(id_b))