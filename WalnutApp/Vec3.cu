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

	__host__ __device__ Vec3 reflect(const Vec3& l, const Vec3& n) {
		return n * dot(n, l) * 2.0f - l;
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
		Vec3 unorm = u;
		for (int i = 0; i < 3; i++) {
			unorm[i] = fminf(1.0, unorm[i]);
			unorm[i] = fmaxf(0.0, unorm[i]);
		}
		int r = (int)(unorm[0] * 255.0);
		int g = (int)(unorm[1] * 255.0);
		int b = (int)(unorm[2] * 255.0);
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