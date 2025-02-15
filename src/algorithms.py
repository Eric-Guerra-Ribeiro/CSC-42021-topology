from typing import Callable
import random

import numpy as np

from src.ball import Ball


def calculate_ball(points:np.ndarray[np.ndarray[float]]) -> Ball:
    if points.ndim == 1:  
        points = points.reshape(1, -1) 
    n, d = points.shape

    if n == 0:
        return Ball()
    elif n == 1:
        return Ball(points[0], 0)
    elif n == 2:
        center = np.mean(points, axis=0) 
        radius = np.linalg.norm(points[0] - center)
        return Ball(center, radius)
    elif n>d+1:
        raise ValueError("There are too many points")
    else:
        p1 = points[0]
        A = 2 * (points[1:] - p1)  
        b = np.sum(points[1:]**2, axis=1) - np.sum(p1**2)  

        center = np.linalg.lstsq(A, b, rcond=None)[0]
        radius = np.linalg.norm(points[0] - center)
        return Ball(center, radius)


def random_permutation(list_:list[int]) -> list[int]:
    size = len(list_)
    for i in range(size - 1):
        j = random.randint(i, size - 1)
        list_[i], list_[j] = list_[j], list_[i]
    return list_


# TODO Maybe delete this, I don't think it works
def seidel(global_set:list[int], func:Callable, boundary_set:list[int], dimension:int) -> tuple[Ball, list[int]]:
    interior_set = []
    critical_set = boundary_set

    # Global set is empty, return ball made by the boundary set
    # Or we have enough points in the boundary set for the problem to be determined
    if not global_set or len(boundary_set) == dimension:
        ball_from_critical_set, _ = func(critical_set, None)
        return ball_from_critical_set, critical_set

    for i in random_permutation(global_set):
        ball_from_critical_set, critical_set_has_changed = func(critical_set, i)
        if critical_set_has_changed:
            ball_from_critical_set, critical_set = seidel(interior_set, func, boundary_set + [i], dimension)
        interior_set.append(i)

    return ball_from_critical_set, critical_set


def min_enclosing_ball_aux(
        interior_set:list[int], calculate_ball:Callable, boundary_set:list[int], dimension:int, points_mapping:np.ndarray[np.ndarray[float]]
    ) -> tuple[Ball, list[int]]:
    # Interior set is empty, return ball made by the boundary set
    # Or we have enough points in the boundary set for the problem to be determined
    if not interior_set or len(boundary_set) == dimension + 1:
        ball = calculate_ball(boundary_set)
        return ball, boundary_set

    # Choose a random point in the interior
    choosen_point_idx = random.randint(0, len(interior_set) - 1)
    choosen_point = interior_set[choosen_point_idx]
    interior_set[choosen_point_idx], interior_set[-1] = interior_set[-1], choosen_point

    # Create ball without this point explicity
    ball, new_boundary_set = min_enclosing_ball_aux(interior_set[:-1], calculate_ball, boundary_set, dimension, points_mapping)

    # Choosen point is not in the ball created without it
    # So it must be in the boundary of the real minimal ball
    if not ball.is_in(points_mapping[interior_set[-1]]):
        ball, new_boundary_set = min_enclosing_ball_aux(interior_set[:-1], calculate_ball, boundary_set + [choosen_point], dimension, points_mapping)

    return ball, new_boundary_set


def min_enclosing_ball(points:np.ndarray[np.ndarray[float]]) -> tuple[Ball, list[int]]:
    def func(points_id:list[int], points:np.ndarray[np.ndarray[float]]) -> tuple[Ball, bool]:
        _, d = points.shape
        selected_points = np.fromiter((points[point_id] for point_id in points_id), dtype=np.dtype((float, d)))
        ball = calculate_ball(selected_points)
        return ball
    n, d = points.shape
    return min_enclosing_ball_aux(list(range(n)), lambda points_id: func(points_id, points), [], d, points)
