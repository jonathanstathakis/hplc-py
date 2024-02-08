from cachier import cachier
from time import sleep
import os
from hplc_py import ROOT

cache_dir = os.path.join(ROOT, "testing_cache", "cache_dir")

@cachier()
def a_cachable_func(arga: str, argb: int):
    sleep(5)
    
    return dict(arga=argb)

if __name__ == "__main__":
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)

    arga="a"
    argb=1
    
    a_cachable_func(arga, argb)


    