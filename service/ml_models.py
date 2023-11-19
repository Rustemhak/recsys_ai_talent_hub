import random
import typing


class Random:
    def predict(self, X) -> typing.List[int]:
        return [random.randint(0, 1000) for _ in range(1000)]
