from __future__ import print_function
from __future__ import division
import numpy as np
import random
import math
import itertools
import pylab
from mpl_toolkits.mplot3d import Axes3D

# user defined options
disk = False                # this parameter defines if we look for Poisson-like distribution on a disk/sphere (center at 0, radius 1) or in a square/box (0-1 on x and y)
repeatPattern = True        # this parameter defines if we look for "repeating" pattern so if we should maximize distances also with pattern repetitions
num_points = 25             # number of points we are looking for
num_iterations = 16         # number of iterations in which we take average minimum squared distances between points and try to maximize them
first_point_zero = disk     # should be first point zero (useful if we already have such sample) or random
iterations_per_point = 64   # iterations per point trying to look for a new point with larger distance
sorting_buckets = 0         # if this option is > 0, then sequence will be optimized for tiled cache locality in n x n tiles (x followed by y)
num_dim = 2                 # 1, 2, 3 dimensional version

def random_point_disk():
    alpha = random.random() * math.pi * 2.0
    radius = math.sqrt(random.random())
    x = math.cos(alpha) * radius
    y = math.sin(alpha) * radius
    return np.array([x,y])

def random_point_sphere():
    theta = random.random() * math.pi * 2.0
    phi = math.acos(2.0 * random.random() - 1.0)
    radius = pow(random.random(), 1.0 / 3.0)
    x = math.cos(theta) * math.sin(phi) * radius
    y = math.sin(theta) * math.sin(phi) * radius
    z = math.cos(phi) * radius
    return np.array([x,y,z])

def random_point_line():
    x = random.random()
    return np.array([x])

def random_point_square():
    x = random.random()
    y = random.random()
    return np.array([x,y])

def random_point_box():
    x = random.random()
    y = random.random()
    z = random.random()
    return np.array([x,y,z])

def first_point():
    if first_point_zero == True:
        return np.array(zero_point)
    return random_point()

# if we only compare it doesn't matter if it's squared
def min_dist_squared_pure(points, point):
    diff = points - np.array([point])
    return np.min(np.einsum('ij,ij->i',diff,diff))

def min_dist_squared_line_repeat(points, point):
    dist = math.sqrt(1.0)
    for x in range(-1,2):
        testing_point = np.array([point-[x]])
        diff = points-testing_point
        dist = min(np.min(np.einsum('ij,ij->i',diff,diff)),dist)
    return dist

def min_dist_squared_repeat(points, point):
    dist = math.sqrt(2.0)
    for y in range(-1,2):
        for x in range(-1,2):
            testing_point = np.array([point-[x,y]])
            diff = points-testing_point
            dist = min(np.min(np.einsum('ij,ij->i',diff,diff)),dist)
    return dist

def min_dist_squared_repeat_3D(points, point):
    dist = math.sqrt(3.0)
    for y in range(-1,2):
        for x in range(-1,2):
            for z in range(-1,2):
                testing_point = np.array([point-[x,y,z]])
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
    best_dist_avg = 0
    for i in range(num_iter):
        points = np.array([first_point()])
        for i in range(num_points-1):
            points = np.append(points, np.array(find_next_point(points),ndmin = 2), axis = 0)
        current_set_dist = 0
        for i in range(num_points):
            dist = min_dist_squared(np.delete(points,i,0), points[i])
            current_set_dist += dist
        if current_set_dist > best_dist_avg:
            best_dist_avg = current_set_dist
            best_point_set = points
    return best_point_set

if num_dim == 3:
    zero_point = [0,0,0]
    if disk == False and repeatPattern == True:
        min_dist_squared = min_dist_squared_repeat_3D
    else:
        min_dist_squared = min_dist_squared_pure
    if disk == True:
        random_point = random_point_sphere
    else:
        random_point = random_point_box
elif num_dim == 2:
    zero_point = [0,0]
    if disk == False and repeatPattern == True:
        min_dist_squared = min_dist_squared_repeat
    else:
        min_dist_squared = min_dist_squared_pure
    if disk == True:
        random_point = random_point_disk
    else:
        random_point = random_point_square
else:
    zero_point = [0]
    if repeatPattern == True:
        min_dist_squared = min_dist_squared_line_repeat
    else:
        min_dist_squared = min_dist_squared_pure
    random_point = random_point_line

points = find_point_set(num_points,num_iterations)

if num_dim == 3:
    if sorting_buckets > 0:
        points_discretized = np.floor(points * [sorting_buckets,-sorting_buckets, sorting_buckets])
        # we multiply in following line by 2 because of -1,1 potential range
        indices_cache_space = np.array(points_discretized[:,2] * sorting_buckets * 4 + points_discretized[:,1] * sorting_buckets * 2 + points_discretized[:,0])
        points = points[np.argsort(indices_cache_space)]
    
    print("// hlsl array")
    print("static const uint SAMPLE_NUM = " + str(num_points) + ";")
    print("static const float3 POISSON_SAMPLES[SAMPLE_NUM] = ")
    print("{")
    for p in points:
        print("float3( " + str(p[0]) + "f, " + str(p[1]) + "f, " + str(p[2]) + "f ), ")
    print("};")
    
    print("// C++ array")
    print("const int SAMPLE_NUM = " + str(num_points) + ";")
    print("const float POISSON_SAMPLES[SAMPLE_NUM][3] = ")
    print("{")
    for p in points:
        print(str(p[0]) + "f, " + str(p[1]) + "f, " + str(p[2]) + "f, ")
    print("};")
    

    ax = pylab.figure(figsize=(10,10)).add_subplot(111, projection='3d')

    if disk == True:
        #less optimal, more readable
        sphere_guide = [[0,0,0]]
        num_guides = 30
        for theta in np.linspace(0, 2.0 * math.pi, num_guides):
            for phi in np.arccos(pylab.linspace(-1, 1.0, num_guides)):
                x = np.cos(theta) * np.sin(phi)
                y = np.sin(theta) * np.sin(phi)
                z = np.cos(phi)   
                sphere_guide = np.append(sphere_guide, np.array([[x,y,z]],ndmin = 2), axis = 0)
        print(sphere_guide)
        ax.plot_wireframe(sphere_guide[1:,0], sphere_guide[1:,1], sphere_guide[1:,2])
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.set_zlim(-1,1)
    elif repeatPattern == True:
        ax.scatter(points[:,0], points[:,1], points[:,2] + 1, c='b')
        ax.scatter(points[:,0], points[:,1] + 1, points[:,2] + 1, c='b')
        ax.scatter(points[:,0] + 1, points[:,1] + 1, points[:,2] + 1, c='b')
        ax.scatter(points[:,0] + 1, points[:,1], points[:,2] + 1, c='b')
        ax.scatter(points[:,0], points[:,1] + 1, points[:,2], c='b')
        ax.scatter(points[:,0] + 1, points[:,1] + 1, points[:,2], c='b')
        ax.scatter(points[:,0] + 1, points[:,1], points[:,2], c='b')
        ax.set_xlim(0,2)
        ax.set_ylim(0,2)
        ax.set_zlim(0,2)
    else:
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_zlim(0,1)


    ax.scatter(points[:,0], points[:,1], points[:,2], c='r')
    pylab.show()    
elif num_dim == 2:
    if sorting_buckets > 0:
        points_discretized = np.floor(points * [sorting_buckets,-sorting_buckets])
        # we multiply in following line by 2 because of -1,1 potential range
        indices_cache_space = np.array(points_discretized[:,1] * sorting_buckets * 2 + points_discretized[:,0])
        points = points[np.argsort(indices_cache_space)]
    
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
        param = np.linspace(0, 2.0 * math.pi, 1000)
        x = np.cos(param)
        y = np.sin(param)
        pylab.plot(x, y, 'b-')    
    elif repeatPattern == True:
        pylab.plot(points[:,0] + 1, points[:,1], 'bo')
        pylab.plot(points[:,0] + 1, points[:,1] + 1, 'bo')
        pylab.plot(points[:,0], points[:,1] + 1, 'bo')
    pylab.plot(points[:,0], points[:,1], 'ro')
    pylab.show()
else:
    if sorting_buckets > 0:
        points_discretized = np.floor(points * [sorting_buckets])
        indices_cache_space = np.array(points_discretized[:,0])
        points = points[np.argsort(indices_cache_space)]
    
    print("// hlsl array")
    print("static const uint SAMPLE_NUM = " + str(num_points) + ";")
    print("static const float POISSON_SAMPLES[SAMPLE_NUM] = ")
    print("{")
    for p in points:
        print(str(p[0]) + "f, ")
    print("};")
    
    print("// C++ array")
    print("const int SAMPLE_NUM = " + str(num_points) + ";")
    print("const float POISSON_SAMPLES[SAMPLE_NUM] = ")
    print("{")
    for p in points:
        print(str(p[0]) + "f, ")
    print("};")
    
    pylab.figure(figsize=(10,2))
    pylab.plot(points[:,0], [0] * num_points, 'ro')
    if repeatPattern == True:
        pylab.plot(points[:,0] + 1, [0] * num_points, 'bo')
    pylab.show()
