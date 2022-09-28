

from typing import Tuple


def foo(a:list, b:dict, c:int) -> Tuple[list, dict, int]:
    
    
    return a, b, c


print(foo([1], {1:"1"}, 1))