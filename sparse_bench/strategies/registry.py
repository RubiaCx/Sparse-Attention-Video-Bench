from sparse_bench.morpher import Strategy
from typing import Callable

class Registry:
    def __init__(self):
        self.data = {}

    def register(self, name: str):
        def wrapper(cls):
            self.data[name] = cls
            return cls
        return wrapper

    def get(self, name: str):
        return self.data[name]
    

STRATEGIES = Registry()
    