from typing import Any

def isArrayLike(x: Any):
    
    if not any(x):
        raise ValueError("x is None")
    
    if not hasattr(x, "__array__"):
        return False
    else:
        return True