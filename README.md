PoissonSamplingGenerator
========================

![Poisson generator screenshot](/poisson.jpg)

Simple Poisson-like distribution generator for various rendering scenarios and sampling (disk, square, grid, repeating grid, rotating disk).
It supports 1D, 2D and 3D variant of distribution to make sampling easier also in volumetric case.

Unlike other simple Poisson generator this one has various options suited specifically for rendering scenarios.
It outputs ready-to-use patterns for both hlsl and C++ code.
It plots pattern on very simple graphs.
Visualization for repeating patterns shows them repeated as well.

Generated sequence has properties of maximizing distance for every next point from previous points in sequence. Therefore you can use partial sequences (for example only half or a few samples based on branching) and have proper sampling function variance. It could be useful for various importance sampling and temporal refinement scenarios.

I also added an option to sort sequence for tiled cache locality if we plan to use a fixed, large number of samples and sample a large area.

### Usage 

There are two ways of using this script:

#### Using GUI
Just launch the script: `python main_gui.py`

#### Using commandline
Just edit the options in main.py and execute script: `python main.py`

##### Options for commandline

Options are edited in code (I use it in Sublime Text and always launch as script, so sorry - no commandline parsing) and are self-describing.

```python
# user defined options
disk = False                # this parameter defines if we look for Poisson-like distribution on a disk/sphere (center at 0, radius 1) or in a square/box (0-1 on x and y)
repeatPattern = True        # this parameter defines if we look for "repeating" pattern so if we should maximize distances also with pattern repetitions
num_points = 25             # number of points we are looking for
num_iterations = 16         # number of iterations in which we take average minimum squared distances between points and try to maximize them
first_point_zero = disk     # should be first point zero (useful if we already have such sample) or random
iterations_per_point = 64   # iterations per point trying to look for a new point with larger distance
sorting_buckets = 0         # if this option is > 0, then sequence will be optimized for tiled cache locality in n x n tiles (x followed by y)
num_dim = 2                 # 1, 2, 3 dimensional version
num_rotations = 1           # number of rotations of disk pattern to check against
```

### Requirements

This simple script requires some scientific Python environment like Anaconda or WinPython. Tested with Anaconda.
Anaconda contains PyQT4, so it should be easy to use with GUI as well. Tested on Windows and Mac OSX.

### Author
Bartlomiej "Bart" Wronski

https://twitter.com/BartWronsk

http://bartwronski.com/
