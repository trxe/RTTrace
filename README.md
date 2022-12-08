# GPU Accelerated Real Time Raytracer

Features implemented:

+ Unity-style camera movement
+ Raytracing with control over levels of recursion from GUI
+ Intersections with spheres, planes and triangles as primitives
+ Simple point lighting
+ AABB intersection checks for acceleration

## Features to implement

- [x] Loading of 3D meshes into scene (FBX binary and OBJ ascii parser)
	- [ ] [Optimizing](https://developer.nvidia.com/blog/introducing-low-level-gpu-virtual-memory-management/) cudaMalloc on change of scene
- [ ] [HLBVH](https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/) acceleration structure
- [ ] Movement of primitives
- [ ] Environment cubemap
- [ ] Area lights
- [ ] Distribution raytracing (Depth of field, anti-aliasing)

## Future upgrades

- [ ] Path tracing
- [ ] Support for BRDFs and BTDFs