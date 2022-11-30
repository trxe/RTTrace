#include <iostream>

#include "cuda_runtime.h"

#include "Util.h"
#include "Vec3.cuh"

#ifndef RAY_H
#define RAY_H

namespace RTTrace {
	struct Ray {
		Ray() = delete;
		__host__ __device__ Ray(Vec3 origin, Vec3 dir) : origin{ origin }, dir{ norm(dir)} {};

		Vec3 origin;
		Vec3 dir;

		__host__ __device__ inline Vec3 point(float t) const {
			return origin + dir * t;
		}

		friend std::ostream& operator<<(std::ostream& os, const Ray& ray);
	};
}

#endif