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
		float t = T_MAX+1;
		Vec3 pos;
		Vec3 norm;
		Vec3 view_dir;
		Vec3 refl_dir;
		// the caller of hit(r, hit) is responsible for adding the surface's index here.
		int surface_index = -1;
	};

	struct MaterialInfo {
		Vec3 ka;
		Vec3 kd;
		Vec3 ks;
		Vec3 krg;
		Vec3 ktg;
		float n = 8;
		float m = 1;
		float mu = 1; // refractive index
	};

	struct SurfaceInfo {
		enum Type { SPHERE, PLANE };
		Type type;
		Vec3 origin;
		Vec3 normal;
		float scale;
		MaterialInfo mat;
	};

	__device__ bool hit_sphere(const Ray& r, const SurfaceInfo& surface, HitInfo& hit);
	__device__ bool hit_plane(const Ray& r, const SurfaceInfo& surface, HitInfo& hit);

}

#endif
