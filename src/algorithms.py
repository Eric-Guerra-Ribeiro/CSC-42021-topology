from typing import Callable
import random

def random_permutation(list_:list[int]) -> list[int]:
    size = len(list_)
    for i in range(size - 1):
        j = random.randint(i, size - 1)
        list_[i], list_[j] = list_[j], list_[i]
    return list_


def seidel(global_set:list[int], func:Callable, possible_set:list[int]) -> list[int]:
    rejected_set = []
    critical_set = possible_set
    
    ball_from_critical_set = None

    for i in random_permutation(global_set):
        ball_from_critical_set, critical_set_has_changed = func(critical_set, i)
        if critical_set_has_changed:
            ball_from_critical_set, critical_set = seidel(rejected_set, func, possible_set + [i])
        rejected_set.append(i)
    
    return ball_from_critical_set, critical_set
