def timed(fn):
    from time import perf_counter
    from functools import wraps

    @wraps(fn)
    def inner(*args, **kwargs):
        start = perf_counter()
        result = fn(*args, *kwargs)
        end = perf_counter()
        elapsed = end - start
        
        args_ = [str(a) for a in args]
        kwargs_ = [ '{0}={1}'.format(k,v) for (k,v) in kwargs.items()]
        all_args = args_ + kwargs_
        args_str = ','.join(all_args)
        print('{0}({1}) took {2: .6f}s to run'.format(fn.__name__, args_str, elapsed))
        return result
    return inner


def dec1(fn):
    def inner():
        print("dec1 is called")
        return fn()
    return inner


def dec2(fn):
    def inner():
        print("dec2 is called")
        return fn()
    return inner


@dec1
@dec2
def my_func():
    print("my func is called")
    
my_func()