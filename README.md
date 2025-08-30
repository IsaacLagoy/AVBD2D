# AVBD2D

## Solver Step Stages

* Broad Collision
* Force Warmstart
* Narrow Collision
* Warmstart Bodies
* Graph Coloring
* Main Solver
* Update Velocities

## TODO
* Broadphase
    - Add SAH BVH
    - Multithread speculative contact pair generation
* Force Warmstart
    - Batch Multithread
    - Concave Collisions
    - Soft Collisions
* Main Solver
    - Vectorize (numpy and numba)
    - Multithread