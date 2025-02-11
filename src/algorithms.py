from typing import Callable
import random
import numpy as np



class data:
    def __init__(self,crutical_set,radius,center,rejected_set):
        self.crutical_set = crutical_set
        self.radius=radius
        self.center=center
        self.rejected_set=rejected_set

def random_permutation(set_:set[int]) -> list[int]:
    permutation = list(set_)
    size = len(permutation)
    for i in range(size - 1):
        j = random.randint(i, size - 1)
        permutation[i], permutation[j] = permutation[j], permutation[i]
    return permutation


def seidel( func:Callable, possible_set:set[int],points:np.ndarray) -> set[int]:
    rejected_set = set()
    crutical_set = possible_set
    selected_points=points[list(crutical_set)]
    radius=func(selected_points).radius
    center=func(selected_points).center

    global_set=set(range(len(points)))

    if not global_set:
        return data(crutical_set, radius, center)
    for i in random_permutation(global_set- crutical_set - rejected_set):
        
        new_selected_points = np.append(selected_points, [points[i]], axis=0) 
        if func(selected_points).radius != func(new_selected_points).radius:
            new_data= seidel(func, crutical_set | {i}, points)
            crutical_set = new_data.crutical_set
            radius = new_data.radius
            center = new_data.center
            rejected_set = new_data.rejected_set
        else: rejected_set = rejected_set | {i}
    
    return data(crutical_set,radius,center,rejected_set)

