import time
from functools import wraps 

def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        value = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[{func.__name__}] 耗时 {elapsed:.4f}s")
        return value
    return wrapper

@timing
def slow_add(x, y):
    time.sleep(0.5)
    return x + y

slow_add(2, 3)
