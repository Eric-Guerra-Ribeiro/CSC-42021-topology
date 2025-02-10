import numpy as np

import src.algorithms as alg
from src.ball import Ball

def one_dimentional_cech_complex_test():

    def calculate_ball(points):
        if len(points) == 0:
            return Ball()
        elif len(points) == 1:
            return Ball(points[0], 0.)
        elif len(points) == 2:
            return Ball(0.5*(points[0] + points[1]), 0.5*abs(points[0] - points[1]))
        else:
            raise ValueError

    def func(points, x):
        ball = calculate_ball(points)
        return ball, not ball.is_in(x)

    
    sample = np.random.randint(low=-100,high=100,size=30,dtype=int)
    min_val = np.min(sample)
    max_val = np.max(sample)

    solution_ball = Ball(0.5*(max_val + min_val), 0.5*(max_val - min_val))
    solution_critical_set = [min_val, max_val]

    sample = list(int(x) for x in sample)


    ball, critical_set = alg.seidel(sample, func, [])
    critical_set.sort()
    

    # Debug

    assert ball.are_the_same(solution_ball)
    assert tuple(solution_critical_set) == tuple(critical_set)
