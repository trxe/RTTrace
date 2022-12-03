#include "cuda_runtime.h"
#include "Vec3.cuh"
#include "Utils.cuh"

namespace RTTrace {
	__host__ __device__ Vec3 operator-(const Vec3& u) {
		return Vec3(-u[0], -u[1], -u[2]);
	}

	__host__ __device__ float dot(const Vec3& u, const Vec3& v) {
		return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
	}

	__host__ __device__ Vec3 norm(const Vec3& u) {
		return u / u.len();
	}

	__host__ __device__ Vec3 cross(const Vec3 &u, const Vec3 &v) {
		return Vec3(u[1] * v[2] - u[2] * v[1],
			u[2] * v[0] - u[0] * v[2],
			u[0] * v[1] - u[1] * v[0]);
	}

	__host__ __device__ abgr_t vec3_to_abgr(const Vec3 &u) {
		abgr_t result;
		float len = u.len();
		Vec3 unorm = (len > 1.0) ? u / len : u;
		int r = (int)(fabs(unorm[0]) * 255.0);
		int g = (int)(fabs(unorm[1]) * 255.0);
		int b = (int)(fabs(unorm[2]) * 255.0);
		abgr_t gb = g << 8;
		abgr_t bb = b << 16;
		result = 0xff000000 | (abgr_t)b << 16 | (abgr_t)g << 8 | (abgr_t)r;
		return result;
	}

	std::ostream& operator<<(std::ostream& os, const Vec3& vec) {
		os << "{";
		for (int i = 0; i < 3; i++) {
			os << vec[i] << " ";
		}
		os << "}";
		return os;
	}
}