import math
import random

import numpy as np
import numpy.linalg as LIN

import src.algorithms as alg
from src.ball import Ball


def n_dimentional_cech_complex_test():
    # 1) For a single point, the center is the point itself, with a radius of 0.

    point = np.array([3.14, -0.5, 2])

    test1 = np.array([point])

    solution1_ball = Ball(point, 0)
    ball1, critical_set1 = alg.min_enclosing_ball(test1)


    assert ball1.are_the_same(solution1_ball)

    # 2) For 2 points, the center is the midpoint, and the radius is half of the distance between the points.
    point1 = np.array([0, -0.5, 2])
    point2 = np.array([3.14, -0.3, -2.71])

    test2 = np.array([point1, point2])

    solution2_ball = Ball(0.5*(point1 + point2), 0.5*LIN.norm(point1 - point2))
    ball2, critical_set2 = alg.min_enclosing_ball(test2)

    assert ball2.are_the_same(solution2_ball)

    # 3) For 3 points with coordinates (-10,0,0), (10,0,0), (0,1,0), the center is (0,0,0) and the radius 10.

    test3 = np.array([
        [-10, 0, 0],
        [10, 0, 0],
        [0, 1, 0]
    ])

    solution3_ball = Ball(np.array([0, 0, 0]), 10)
    ball3, critical_set3 = alg.min_enclosing_ball(test3)

    assert ball3.are_the_same(solution3_ball)

    # 4) For 3 points with coordinates (-5,0,0), (3,-4,0), (3,4,0), the center is (0,0,0) and the radius 5.

    test4 = np.array([
        [-5, 0, 0],
        [3, -4, 0],
        [3, 4, 0]
    ])

    solution4_ball = Ball(np.array([0, 0, 0]), 5)
    ball4, critical_set4 = alg.min_enclosing_ball(test4)
    
    assert ball4.are_the_same(solution4_ball)

    # 5) For 4 points with coordinates (5,0,1), (-1,-3,4), (-1,-4,-3), (- 1,4,-3), the center is (0,0,0) and the radius √26 ≈ 5.09902

    test5 = np.array([[5, 0, 1],
                     [-1, -3, 4],
                     [-1, -4, -3],
                     [-1, 4, -3]])

    solution5_ball = Ball(np.array([0, 0, 0]), np.sqrt(26))
    ball5, critical_set5 = alg.min_enclosing_ball(test5)

    assert ball5.are_the_same(solution5_ball)

    # 6) You can then experiment with adding more points inside the sphere and changing the order of the points, which should not change the MEB.

    test6 = np.array([[-1, -3, 4],
                     [-1, 4, -3],
                     [-1, -4, -3],
                     [5, 0, 1]])

    n, d = test6.shape

    N = 100

    new_points = np.fromiter(
        ([r*math.sin(theta)*math.cos(phi), r*math.sin(theta)*math.sin(phi), r*math.cos(theta)]
         for r, theta, phi
         in zip(np.random.uniform(0, 4, N - n), np.random.uniform(0, np.pi, N - n), np.random.uniform(-np.pi, np.pi, N - n)))
        , dtype=np.dtype((float, d))
    )

    test6_full = np.concatenate([test6, new_points], axis=0)

    ball6, critical_set6 = alg.min_enclosing_ball(test6_full)

    assert ball6.are_the_same(solution5_ball)


def enumerate_simplexes_ck_test():
    
    test = np.array([[5, 0, 1],
                     [-1, -3, 4],
                     [-1, -4, -3],
                     [-1, 4, -3]])

    print("\n".join(sorted(alg.enumerate_simplexes_ck(test, 4), key=len)))


def enumerate_simplexes_ckl_test():
    
    test = np.array([[5, 0, 1],
                     [-1, -3, 4],
                     [-1, -4, -3],
                     [-1, 4, -3]])

    print("\n".join(sorted(alg.enumerate_simplexes_ckl(test, 4, 4), key=len)))

