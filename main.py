
import numpy as np

import src.algorithms as alg

def englob_line(set_):
    def func(set__):
        if len(set__) == 0:
            return -1
        elif len(set__) == 1:
            return 0
        elif len(set__) == 2:
            return max(set__) - min(set__)
        elif len(set__) == 3:
            return max(set__) - min(set__)+1786878
        else:
            raise ValueError
    crit = alg.seidel(set_, func, set())

    assert len(crit) <= 2
    return crit

sample = np.random.randint(low=-100,high=100,size=100,dtype=int)
min_val = np.min(sample)
max_val = np.max(sample)
test = set(int(x) for x in sample)

print(englob_line(test))
print(f"{min_val}, {max_val}: {max_val - min_val}")
