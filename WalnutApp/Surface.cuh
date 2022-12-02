#ifndef SURFACE_H
#define SURFACE_H

#include <memory>
#include <vector>

#include "cuda_runtime.h"
#include "Ray.cuh"
#include "Vec3.cuh"

namespace RTTrace {

#define T_MAX 1000000
#define T_EPSILON 0.000001

	struct HitInfo {
		float t;
		Vec3 pos;
		Vec3 norm;
		// the caller of hit(r, hit) is responsible for adding the surface's index here.
		int surface_index;
	};

	/*
	class Surface {
	public:
		__device__ virtual bool hit(const Ray& r, HitInfo* hit) const = 0;
	};

	class Sphere : public Surface {
	public:
		Sphere(Vec3 origin, float radius) : origin(origin), radius(radius) {};
		__device__ virtual bool hit(const Ray& ray, HitInfo* hit) const override;
	private:
		Vec3 origin;
		float radius;
	};

	class Plane : public Surface {
	public:
		Plane(Vec3 normal, Vec3 origin) : origin(origin), normal(normal) {
			n_dot_point = dot(origin, normal);
		};
		__device__ virtual bool hit(const Ray& ray, HitInfo* hit) const override;
	private:
		Vec3 origin;
		Vec3 normal;
		float n_dot_point;
	};
	*/

}

#endif
