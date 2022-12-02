#ifndef SURFACE_H
#define SURFACE_H

#include <memory>
#include <vector>

#include "cuda_runtime.h"
#include "Ray.cuh"
#include "Vec3.cuh"

namespace RTTrace {

#define T_MAX 1000000
#define T_EPSILON 0.01

	struct HitInfo {
		bool is_hit = false;
		float t = -1;
		Vec3 pos;
		Vec3 norm;
		// the caller of hit(r, hit) is responsible for adding the surface's index here.
		int surface_index = -1;
	};

	struct SurfaceInfo {
		enum Type { SPHERE, PLANE };
		Type type;
		Vec3 origin;
		Vec3 normal;
		float scale;
	};

	__device__ bool hit_sphere(const Ray& r, const SurfaceInfo& surface, HitInfo& hit);
	__device__ bool hit_plane(const Ray& r, const SurfaceInfo& surface, HitInfo& hit);

}

#endif
