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

	struct AABB {
		bool active = false;
		Vec3 minw;
		Vec3 maxw;
	};

	struct SurfaceInfo {
		enum Type { SPHERE, PLANE, TRIANGLE };
		Type type;
		Vec3 origin; // PLANE
		Vec3 normal; //PLANE
		float scale; // SPHERE
		Vec3 points[3]{ origin, origin, origin }; // TRIANGLE
		MaterialInfo mat;
		AABB bound;
	};

	__host__ bool init_bound(SurfaceInfo& surface);

	__device__ bool hit_bound(const Ray& r, const SurfaceInfo& surface);
	__device__ bool hit_sphere(const Ray& r, const SurfaceInfo& surface, HitInfo& hit);
	__device__ bool hit_plane(const Ray& r, const SurfaceInfo& surface, HitInfo& hit);
	__device__ bool hit_triangle(const Ray& r, const SurfaceInfo& surface, HitInfo& hit);

}

#endif
