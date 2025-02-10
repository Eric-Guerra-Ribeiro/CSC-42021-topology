from __future__ import annotations
import math
from typing import Optional

import numpy as np
import numpy.linalg as LIN

from src.constants import EPSILON_PRECISION

class Ball:

    def __init__(self, center:Optional[np.ndarray[float]]=None, radius:Optional[float]=None) -> None:
        self.center = center
        self.radius = radius
    
    def is_in(self, point:np.ndarray[float]) -> bool:
        if self.center is None or self.radius is None:
            return False
        return LIN.vector_norm(point - self.center, ord=2) <= self.radius

    def are_the_same(self, ball:Ball):
        # We consider non-existing balls as unique
        if self.center is None or ball.center is None:
            return False
        if self.radius is None or ball.radius is None:
            return False

        return (math.fabs(ball.radius - self.radius) < EPSILON_PRECISION
                and LIN.vector_norm(ball.center - self.center) < EPSILON_PRECISION)

    def __str__(self) -> str:
        return f"(({self.center}) {self.radius})"

    def __repr__(self) -> str:
        return self.__str__()
