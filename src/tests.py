import numpy as np
import numpy.linalg as LIN

import src.algorithms as alg
from src.ball import Ball


def n_dimentional_cech_complex_test():

    def func(points_id:list[int], points:np.ndarray[np.ndarray[float]]) -> tuple[Ball, bool]:
        _, d = points.shape
        selected_points = np.fromiter((points[point_id] for point_id in points_id), dtype=np.dtype((float, d)))
        ball = alg.calculate_ball(selected_points)
        return ball

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
