from typing import Any
import time

def timer(op_str: str, on: bool=False) -> Any:
    def time_dec(func) -> Any:
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            if on: print(f'{op_str}: {(time.perf_counter() - start) * 1000:.3f}ms')
            return result
        return wrapper
    return time_dec