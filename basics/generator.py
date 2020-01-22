def simple_gen():
    for i in range(3):
        yield i

g = simple_gen()
next(g) #0
next(g) #1
next(g) #2


def gen_cube(n):
    for x in range(n):
        yield x**3
#返回generator object        
gen_cube(4) #<generator object gen_cube at 0x10567b150>
#pull value from gen_cube
print(list(gen_cube(4))) #