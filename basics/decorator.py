import functools

def my_decorator1(func): 
    @functools.wraps(func)
    def wrapper(*args, **kwargs): 
        print('wrapper of decorator1') 
        func(*args, **kwargs) 
    return wrapper

def my_decorator2(func): 
    @functools.wraps(func)
    def wrapper(*args, **kwargs): 
        print('wrapper of decorator2') 
        func(*args, **kwargs) 
    return wrapper
        
@my_decorator1
def func1(msg): 
    print(msg)

@my_decorator1
@my_decorator2
def func2(arg1, arg2):
    print(arg1, arg2)
    
func1("func1")
func2("func2","123")

def repeat(num):
    def my_decorator(func): 
        @functools.wraps(func)
        def wrapper(*args, **kwargs): 
            for i in range(num):
                print('wrapper of decorator') 
                func(*args, **kwargs) 
        return wrapper
    return my_decorator

@repeat(4)
def func3(msg):
    print(msg)

func3("func3")    

class MyClass:
    pass

print(type(MyClass))