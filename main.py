from __future__ import print_function
import numpy as np
import math
import matplotlib.pyplot as plt
import poisson

def main():
    # user defined options
    disk = False                # this parameter defines if we look for Poisson-like distribution on a disk/sphere (center at 0, radius 1) or in a square/box (0-1 on x and y)
    repeatPattern = True        # this parameter defines if we look for "repeating" pattern so if we should maximize distances also with pattern repetitions
    num_points = 8              # number of points we are looking for
    num_iterations = 4          # number of iterations in which we take average minimum squared distances between points and try to maximize them
    first_point_zero = disk     # should be first point zero (useful if we already have such sample) or random
    iterations_per_point = 128  # iterations per point trying to look for a new point with larger distance
    sorting_buckets = 0         # if this option is > 0, then sequence will be optimized for tiled cache locality in n x n tiles (x followed by y)
    num_dim = 2                 # 1, 2, 3 dimensional version
    num_rotations = 1           # number of rotations of pattern to check against


    poisson_generator = poisson.PoissonGenerator(num_dim, disk, repeatPattern, first_point_zero)
    points = poisson_generator.find_point_set(num_points, num_iterations, iterations_per_point, num_rotations)
    points = poisson_generator.cache_sort(points, sorting_buckets)
    print(poisson_generator.format_points_string(points))

    fig = plt.figure(figsize=(10,10))
    poisson_generator.generate_ui(fig, points)
    plt.show()

if __name__ == '__main__':
    main()