PoissonSamplingGenerator
========================

Simple Poisson-like distribution generator for various rendering scenarios and sampling (disk, square, grid, repeating grid).

Unlike other simple Poisson generator this one has various options suited specifically for rendering scenarios.
It outputs ready-to-use patterns for both hlsl and C++ code.
It plots pattern on a very simple graphs.

### Usage 

Just edit the options and execute script: `python poisson.py`

### Options

Options are edited in code (I use it in Sublime Text and always launch as script, so sorry - no commandline parsing) and are self-describing.

```python
# user defined options
disk = False                # this parameter defines if we look for Poisson-like distribution on a disk (center at 0, radius 1) or in a square (0-1 on x and y)
squareRepeatPattern = True  # this parameter defines if we look for "repeating" pattern so if we should maximize distances also with pattern repetitions
num_points = 25             # number of points we are looking for
num_iterations = 16         # number of iterations in which we take average minimum squared distances between points and try to maximize them
first_point_zero = disk     # should be first point zero (useful if we already have such sample) or random
iterations_per_point = 64   # iterations per point trying to look for a new point with larger distance
```

### Requirements

This simple script requires some scientific Python environment like Anaconda or WinPython. Tested with Anaconda.

### Author
Bartlomiej "Bart" Wronski

https://twitter.com/BartWronsk

http://bartwronski.com/
