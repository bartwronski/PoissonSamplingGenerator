from __future__ import division
import numpy as np
import math
import scipy.spatial.distance
from mpl_toolkits.mplot3d import Axes3D

def random_point_disk(num_points = 1):
    alpha = np.random.random(num_points) * math.pi * 2.0
    radius = np.sqrt(np.random.random(num_points))
    x = np.cos(alpha) * radius
    y = np.sin(alpha) * radius
    return np.dstack((x,y))[0]

def random_point_sphere(num_points = 1):
    theta = np.random.random(num_points) * math.pi * 2.0
    phi = np.arccos(2.0 * np.random.random(num_points) - 1.0)
    radius = pow(np.random.random(num_points), 1.0 / 3.0)
    x = np.cos(theta) * np.sin(phi) * radius
    y = np.sin(theta) * np.sin(phi) * radius
    z = np.cos(phi) * radius
    return np.dstack((x,y,z))[0]

def random_point_line(num_points = 1):
    x = np.random.random(num_points)
    return np.reshape(x, (num_points,1))

def random_point_square(num_points = 1):
    x = np.random.random(num_points)
    y = np.random.random(num_points)
    return np.dstack((x,y))[0]

def random_point_box(num_points = 1):
    x = np.random.random(num_points)
    y = np.random.random(num_points)
    z = np.random.random(num_points)
    return np.dstack((x,y,z))[0]

# if we only compare it doesn't matter if it's squared
def min_dist_squared(points, point):
    diff = points - np.array([point])
    return np.min(np.einsum('ij,ij->i',diff,diff))

class PoissonGenerator:
    def __init__(self, num_dim, disk, repeatPattern, first_point_zero):
        self.first_point_zero = first_point_zero
        self.disk = disk
        self.num_dim = num_dim
        self.repeatPattern = repeatPattern and disk == False
        self.num_perms = (3 ** self.num_dim) if self.repeatPattern else 1

        if num_dim == 3:
            self.zero_point = [0,0,0]
            if disk == True:
                self.random_point = random_point_sphere
            else:
                self.random_point = random_point_box
        elif num_dim == 2:
            self.zero_point = [0,0]
            if disk == True:
                self.random_point = random_point_disk
            else:
                self.random_point = random_point_square
        else:
            self.zero_point = [0]
            self.random_point = random_point_line

    def first_point(self):
        if self.first_point_zero == True:
            return np.array(self.zero_point)
        return self.random_point(1)[0]

    def find_next_point(self, current_points, iterations_per_point):
        best_dist = 0
        best_point = []
        random_points = self.random_point(iterations_per_point)
        for new_point in random_points:
            dist = min_dist_squared(current_points, new_point)
            if dist > best_dist:
                best_dist = dist
                best_point = new_point
        return best_point

    def permute_point(self, point):
        out_array = np.array(point,ndmin = 2)
        if not self.repeatPattern:
            return out_array

        if self.num_dim == 3:
            for z in range(-1,2):
                for y in range(-1,2):
                    for x in range(-1,2):
                        if y != 0 or x != 0 or z != 0:
                            perm_point = point+[x,y,z]
                            out_array = np.append(out_array, np.array(perm_point,ndmin = 2), axis = 0 )
        elif self.num_dim == 2:            
            for y in range(-1,2):
                for x in range(-1,2):
                    if y != 0 or x != 0:
                        perm_point = point+[x,y]
                        out_array = np.append(out_array, np.array(perm_point,ndmin = 2), axis = 0 )
        else:
            for x in range(-1,2):
                if x != 0:
                    perm_point = point+[x]
                    out_array = np.append(out_array, np.array(perm_point,ndmin = 2), axis = 0 )

        return out_array

    def find_point_set(self, num_points, num_iter, iterations_per_point, rotations, progress_notification = None):
        best_point_set = []
        best_dist_avg = 0
        self.rotations = 1
        if self.disk and self.num_dim == 2:
            rotations = max(rotations, 1)
            self.rotations = rotations

        for i in range(num_iter):
            if progress_notification != None:
                progress_notification(i / num_iter)
            points = self.permute_point(self.first_point())

            for i in range(num_points-1):
                next_point = self.find_next_point(points, iterations_per_point)
                points = np.append(points, self.permute_point(next_point), axis = 0)

            current_set_dist = 0

            if rotations > 1:
                points_permuted = np.copy(points)
                for rotation in range(1, rotations):
                    rot_angle = rotation * math.pi * 2.0 / rotations
                    s, c = math.sin(rot_angle), math.cos(rot_angle)
                    rot_matrix = np.matrix([[c, -s], [s, c]])
                    points_permuted = np.append(points_permuted, np.array(np.dot(points, rot_matrix)), axis = 0)
                current_set_dist = np.min(scipy.spatial.distance.pdist(points_permuted))
            else:
                current_set_dist = np.min(scipy.spatial.distance.pdist(points))

            if current_set_dist > best_dist_avg:
                best_dist_avg = current_set_dist
                best_point_set = points
        return best_point_set[::self.num_perms,:]

    def cache_sort(self, points, sorting_buckets):
        if sorting_buckets < 1:
            return points
        if self.num_dim == 3:
            points_discretized = np.floor(points * [sorting_buckets,-sorting_buckets, sorting_buckets])
            indices_cache_space = np.array(points_discretized[:,2] * sorting_buckets * 4 + points_discretized[:,1] * sorting_buckets * 2 + points_discretized[:,0])
            points = points[np.argsort(indices_cache_space)]
        elif self.num_dim == 2:
            points_discretized = np.floor(points * [sorting_buckets,-sorting_buckets])
            indices_cache_space = np.array(points_discretized[:,1] * sorting_buckets * 2 + points_discretized[:,0])
            points = points[np.argsort(indices_cache_space)]        
        else:
            points_discretized = np.floor(points * [sorting_buckets])
            indices_cache_space = np.array(points_discretized[:,0])
            points = points[np.argsort(indices_cache_space)]
        return points

    def format_points_string(self, points):
        types_hlsl = ["float", "float2", "float3"]

        points_str_hlsl = "// hlsl array\n"
        points_str_hlsl += "static const uint SAMPLE_NUM = " + str(points.size // self.num_dim) + ";\n"
        points_str_hlsl += "static const " + types_hlsl[self.num_dim-1] + " POISSON_SAMPLES[SAMPLE_NUM] = \n{ \n"

        points_str_cpp = "// C++ array\n"
        points_str_cpp += "const int SAMPLE_NUM = " + str(points.size // self.num_dim) + ";\n"
        points_str_cpp += "const float POISSON_SAMPLES[SAMPLE_NUM][" + str(self.num_dim) + "] = \n{ \n"

        if self.num_dim == 3:
            for p in points:
                points_str_hlsl += "float3( " + str(p[0]) + "f, " + str(p[1]) + "f, " + str(p[2]) + "f ), \n"
                points_str_cpp += str(p[0]) + "f, " + str(p[1]) + "f, " + str(p[2]) + "f, \n"
        elif self.num_dim == 2:
            for p in points:
                points_str_hlsl += "float2( " + str(p[0]) + "f, " + str(p[1]) + "f ), \n"
                points_str_cpp += str(p[0]) + "f, " + str(p[1]) + "f, \n"
        else:
            for p in points:
                points_str_hlsl += str(p[0]) + "f, \n"
                points_str_cpp += str(p[0]) + "f, \n"

        points_str_hlsl += "};\n\n"
        points_str_cpp += "};\n\n"

        return points_str_hlsl + points_str_cpp

    def generate_ui(self, fig, points, highlightFirst = 0):
        num_points = points.size // self.num_dim

        if self.num_dim == 3:
            ax = fig.add_subplot(111, projection='3d')
            if self.disk == True:
                #less optimal, more readable
                sphere_guide = [[0,0,0]]
                num_guides = 30
                for theta in np.linspace(0, 2.0 * math.pi, num_guides):
                    for phi in np.arccos(np.linspace(-1, 1.0, num_guides)):
                        x = np.cos(theta) * np.sin(phi)
                        y = np.sin(theta) * np.sin(phi)
                        z = np.cos(phi)   
                        sphere_guide = np.append(sphere_guide, np.array([[x,y,z]],ndmin = 2), axis = 0)
                ax.plot_wireframe(sphere_guide[1:,0], sphere_guide[1:,1], sphere_guide[1:,2])
                ax.set_xlim(-1,1)
                ax.set_ylim(-1,1)
                ax.set_zlim(-1,1)
            elif self.repeatPattern == True:
                ax.scatter(points[:,0], points[:,1], points[:,2] + 1, c='b')
                ax.scatter(points[:,0], points[:,1] + 1, points[:,2] + 1, c='b')
                ax.scatter(points[:,0] + 1, points[:,1] + 1, points[:,2] + 1, c='b')
                ax.scatter(points[:,0] + 1, points[:,1], points[:,2] + 1, c='b')
                ax.scatter(points[:,0], points[:,1] + 1, points[:,2], c='b')
                ax.scatter(points[:,0] + 1, points[:,1] + 1, points[:,2], c='b')
                ax.scatter(points[:,0] + 1, points[:,1], points[:,2], c='b')
                
                a = np.linspace(0, 2.0, 3)
                b = np.linspace(0, 2.0, 3)
                a, b = np.meshgrid(a,b)
                ax.plot_wireframe(a, b, 1.0)
                ax.plot_wireframe(a, 1.0, b)
                ax.plot_wireframe(1.0, a, b)
                
                ax.set_xlim(0,2)
                ax.set_ylim(0,2)
                ax.set_zlim(0,2)

            else:
                ax.set_xlim(0,1)
                ax.set_ylim(0,1)
                ax.set_zlim(0,1)

            ax.scatter(points[highlightFirst:,0], points[highlightFirst:,1], points[highlightFirst:,2], c='g')
            ax.scatter(points[:highlightFirst,0], points[:highlightFirst,1], points[:highlightFirst,2], c='r')
        elif self.num_dim == 2:
            ax = fig.add_subplot(111)
            if self.disk == True:
                param = np.linspace(0, 2.0 * math.pi, 1000)
                x = np.cos(param)
                y = np.sin(param)
                ax.plot(x, y, 'b-')    
            elif self.repeatPattern == True:
                ax.plot(points[:,0] + 1, points[:,1], 'bo')
                ax.plot(points[:,0] + 1, points[:,1] + 1, 'bo')
                ax.plot(points[:,0], points[:,1] + 1, 'bo')
            if self.disk == False:
                param = np.linspace(0, 2.0, 100)
                ax.plot(param, [1] * 100, 'k')
                ax.plot([1] * 100, param, 'k')
            for rotation in range(1,self.rotations):
                rot_angle = rotation * math.pi * 2.0 / self.rotations
                s, c = math.sin(rot_angle), math.cos(rot_angle)
                rot_matrix = np.matrix([[c, -s], [s, c]])
                points_permuted = np.array(np.dot(points, rot_matrix))
                ax.plot(points_permuted[:,0], points_permuted[:,1], 'bo')
            ax.plot(points[:highlightFirst,0], points[:highlightFirst,1], 'go')
            ax.plot(points[highlightFirst:,0], points[highlightFirst:,1], 'ro')
        else:
            ax = fig.add_subplot(111)
            ax.plot(points[:highlightFirst,0], [0] * num_points, 'go')
            ax.plot(points[highlightFirst:,0], [0] * num_points, 'ro')
            if self.repeatPattern == True:
                ax.plot(points[:,0] + 1, [0] * num_points, 'bo')
