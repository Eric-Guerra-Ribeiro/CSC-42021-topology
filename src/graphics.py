#!/usr/bin/env python3
import re

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.pyplot as plt

def convert_enumeration_to_simplex(simplex_str:str):
    """
    Converts the enumeration as a list of string with the information
    '(point0 point1 point2) -> [filtration]', where we may have up to three points that are integers
    and filtration is the radius.
    """
    points_regex = re.search(r"\(([\d\s]+)\)", simplex_str)
    return list(map(int, points_regex.group(1).split()))

def plot3d(simplexes_list, points, title, filename=None):
    simplexes = [[], [], []]
    for simplex in simplexes_list:
        k = len(simplex) - 1
        if k <= 2:
            simplexes[k].append([int(i) for i in simplex])

    fig = plt.figure()

    #ax = fig.gca(projection='3d') # old matplotlib
    ax = fig.add_subplot(projection='3d')
    # Plot triangles
    ax.plot_trisurf(points[:,0], points[:,1], points[:,2], triangles=simplexes[2])
    # Plot points
    points2 = points[np.array(simplexes[0]).reshape(-1)]
    ax.scatter3D(points2[:,0], points2[:,1], points2[:,2])
    # Plot edges
    ax.add_collection3d(Line3DCollection(lines=[points[e] for e in simplexes[1]]))

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if filename:
        plt.savefig(f'{filename}.pdf')
    plt.title(title)
    plt.show()

