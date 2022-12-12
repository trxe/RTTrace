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
		float l = u.len();
		if (l == 0) return u;
		return u / u.len();
	}

	__host__ __device__ float det(const Vec3& c0, const Vec3& c1, const Vec3& c2) {
		float d = 0;
		d += c0[0] * (c1[1] * c2[2] - c1[2] * c2[1]);
		d -= c1[0] * (c0[1] * c2[2] - c0[2] * c2[1]);
		d += c2[0] * (c0[1] * c1[2] - c0[2] * c1[1]);
		return d;
	}

	__host__ __device__ Vec3 cross(const Vec3 &u, const Vec3 &v) {
		return Vec3(u[1] * v[2] - u[2] * v[1],
			u[2] * v[0] - u[0] * v[2],
			u[0] * v[1] - u[1] * v[0]);
	}

	__host__ __device__ Vec3 clamp(const Vec3& u, const Vec3& min, const Vec3& max) {
		Vec3 uclamp = u;
		for (int i = 0; i < 3; i++) {
			uclamp[i] = fminf(max[i], uclamp[i]);
			uclamp[i] = fmaxf(min[i], uclamp[i]);
		}
		return uclamp;
	}

	__host__ __device__ Vec3 vmin(const Vec3& u, const Vec3& v) {
		Vec3 omin = u;
		for (int i = 0; i < 3; i++) {
			omin[i] = fminf(u[i], v[i]);
		}
		return omin;
	}

	__host__ __device__ Vec3 vmax(const Vec3& u, const Vec3& v) {
		Vec3 omax = u;
		for (int i = 0; i < 3; i++) {
			omax[i] = fmaxf(u[i], v[i]);
		}
		return omax;
	}

	/**
	 * Clamps vector to a color ranging between WHITE and BLACK.
	 * 
	 * \param u Unclamped Vector
	 * \return Color after clamping
	 */
	__host__ __device__ Vec3 clamp_color(const Vec3& u) {
		return clamp(u, Vec3(), Vec3(1.0, 1.0, 1.0));
	}

	/**
	 * Finds the Vec3 of parameters t yielding the interpolated u between min and max.
	 * 
	 * \param u Resultant interpolated vector
	 * \param min Min vector
	 * \param max Max vector
	 * \return Vec3 of parameters such that lerp(t, min, max) is u
	 */
	__host__ __device__ Vec3 inverse_lerp(const Vec3& u, const Vec3& min, const Vec3& max) {
		Vec3 t{};
		for (int i = 0; i < 3; i++) {
			if (min[i] == max[i]) t[i] = NAN;
			t[i] = (u[i] - min[i]) / (max[i] - min[i]);
		}
		return t;
	}

	/**
	 * Generates morton code required for LBVH generation.
	 * 
	 * \param pos World space coordinates of centroid of object.
	 * \param global_min World space coordinates of the minimum point on the global AABB.
	 * \param global_max World space coordinates of the maximum point on the global AABB.
	 */
	__host__ __device__ abgr_t vec3_to_abgr(const Vec3 &u) {
		abgr_t result;
		Vec3 uclamp = clamp_color(u);
		int r = (int)(uclamp[0] * 255.0);
		int g = (int)(uclamp[1] * 255.0);
		int b = (int)(uclamp[2] * 255.0);
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

	__host__ __device__ uint32_t generate_morton_code(const Vec3& pos, const Vec3& global_min, const Vec3& global_max) {
		uint32_t morton = 0x0;
		Vec3 rel_pos = inverse_lerp(pos, global_min, global_max);
		for (int i = 0; i < 3; i++) {
			uint32_t v = fminf(fmaxf(0.0, rel_pos[i] * 1024.0), 1023.0);
			uint32_t vv = left_shift_3(v);
			morton += vv << (2 - i);
		}
		return morton;
	}
}