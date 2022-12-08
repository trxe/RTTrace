#ifndef DEMO_H
#define DEMO_H

#include "Surface.cuh"
#include "Light.cuh"

namespace RTTrace {
	__host__ void load_demo_scene(std::vector<SurfaceInfo>& surfaces, std::vector<LightInfo>& lights, AABB& global_bound);
}

#endif // !DEMO_H

