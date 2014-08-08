from __future__ import print_function
from __future__ import division
import numpy as np
import random
import math
import itertools
import pylab

# user defined options
disk = False                # this parameter defines if we look for Poisson-like distribution on a disk (center at 0, radius 1) or in a square (0-1 on x and y)
squareRepeatPattern = True  # this parameter defines if we look for "repeating" pattern so if we should maximize distances also with pattern repetitions
num_points = 25             # number of points we are looking for
num_iterations = 16         # number of iterations in which we take average minimum squared distances between points and try to maximize them
first_point_zero = disk     # should be first point zero (useful if we already have such sample) or random
iterations_per_point = 64   # iterations per point trying to look for a new point with larger distance

def random_point_disk():
    alpha = random.random() * math.pi * 2.0
    radius = math.sqrt(random.random())
    x = math.cos(alpha) * radius
    y = math.sin(alpha) * radius
    return np.array([x,y])

def random_point_square():
    x = random.random()
    y = random.random()
    return np.array([x,y])

def first_point():
    if first_point_zero == True:
        return np.array([0,0])
    elif disk == True:
        return random_point_disk()
    else:
        return random_point_square()

# if we only compare it doesn't matter if it's squared
def min_dist_squared_pure(points, point):
    diff = points - np.array([point])
    return np.min(np.einsum('ij,ij->i',diff,diff))

def min_dist_squared_repeat(points, point):
    dist = math.sqrt(2)
    for y in range(-1,2):
        for x in range(-1,2):
            testing_point = np.array([point-[x,y]])
            diff = points-testing_point
            dist = min(np.min(np.einsum('ij,ij->i',diff,diff)),dist)
    return dist

def find_next_point(current_points):
    best_dist = 0
    best_point = []
    for i in range(iterations_per_point):
        new_point = random_point()
        dist = min_dist_squared(current_points, new_point)
        if dist > best_dist:
            best_dist = dist
            best_point = new_point
    return best_point


def find_point_set(num_points, num_iter):
    best_point_set = []
    best_dist_avg = num_points*math.sqrt(2.0)
    for i in range(num_iter):
        points = np.array([first_point()])
        for i in range(num_points-1):
            points = np.append(points, np.array(find_next_point(points),ndmin = 2), axis = 0)
        current_set_dist = 0
        for i in range(num_points):
            dist = min_dist_squared(np.delete(points,i,0), points[i])
            current_set_dist += dist
        if current_set_dist < best_dist_avg:
            best_dist_avg = current_set_dist
            best_point_set = points
    return best_point_set

if disk == True:
    random_point = random_point_disk
else:
    random_point = random_point_square

if disk == False and squareRepeatPattern == True:
    min_dist_squared = min_dist_squared_repeat
else:
    min_dist_squared = min_dist_squared_pure

points = find_point_set(num_points,num_iterations)

print("// hlsl array")
print("static const uint SAMPLE_NUM = " + str(num_points) + ";")
print("static const float2 POISSON_SAMPLES[SAMPLE_NUM] = ")
print("{")
for p in points:
    print("float2( " + str(p[0]) + "f, " + str(p[1]) + "f ), ")
print("};")

print("// C++ array")
print("const int SAMPLE_NUM = " + str(num_points) + ";")
print("const float POISSON_SAMPLES[SAMPLE_NUM][2] = ")
print("{")
for p in points:
    print(str(p[0]) + "f, " + str(p[1]) + "f, ")
print("};")

pylab.figure(figsize=(10,10))
if disk == True:
    param = pylab.linspace(0, 2.0 * math.pi, 1000)
    x = pylab.cos(param)
    y = pylab.sin(param)
    pylab.plot(x, y, 'b-')    
pylab.plot(points[:,0], points[:,1], 'go')
pylab.show()
