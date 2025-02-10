
import numpy as np

import src.algorithms as alg


def func(points):
    points = np.array(points)  
    n, d = points.shape  

    if points.size==0:
        return -1
    elif n==1:
        return 0
    elif n>d+1:
        raise ValueError("There are too many points")

    p1 = points[0]
    A = 2 * (points[1:] - p1)  
    b = np.sum(points[1:]**2, axis=1) - np.sum(p1**2)  


    center = np.linalg.lstsq(A, b, rcond=None)[0] 
    radius = np.linalg.norm(points[0] - center)

    return radius
 

test = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 ])

print(func(test))

# sample = np.random.randint(low=-100,high=100,size=100,dtype=int)
# min_val = np.min(sample)
# max_val = np.max(sample)
# test = set(int(x) for x in sample)

# print(englob_line(test))
# print(f"{min_val}, {max_val}: {max_val - min_val}")



#  if len(set__) == 0:
#             return -1
#         elif len(set__) == 1:
#             return 0
#         elif len(set__) == 2:
#             return max(set__) - min(set__)
#         elif len(set__) == 3:
#             return max(set__) - min(set__)
#         else:
#             raise ValueError

      
    # crit = alg.seidel(points, func, set())

    # assert len(crit) <= 2
    # return crit