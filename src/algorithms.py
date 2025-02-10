from typing import Callable
import random

def random_permutation(set_:set[int]) -> list[int]:
    permutation = list(set_)
    size = len(permutation)
    for i in range(size - 1):
        j = random.randint(i, size - 1)
        permutation[i], permutation[j] = permutation[j], permutation[i]
    return permutation


def seidel(global_set:set[int], func:Callable, possible_set:set[int]) -> set[int]:
    rejected_set = set()
    critical_set = possible_set

    for i in random_permutation(global_set):
        if func(critical_set) != func(critical_set | {i}):
            critical_set = seidel(rejected_set, func, possible_set | {i})
        rejected_set = rejected_set | {i}
    
    return critical_set
