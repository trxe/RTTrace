#ifndef VEC3_H
#define VEC3_H

#include <math.h>
#include <iostream>

#include "cuda_runtime.h"
#include "Utils.cuh"

namespace RTTrace {
	/**
	 * Vec3 class to be used on the device.
	 */
	struct Vec3 {
	public:
		__host__ __device__ Vec3() : a{ 0,0,0 } {};
		__host__ __device__ Vec3(const float* e) : a{ e[0], e[1], e[2]} {};
		__host__ __device__ Vec3(float a0, float a1, float a2) : a{ a0,a1,a2 } {};

		__host__ __device__ float x() { return a[0]; }
		__host__ __device__ float y() { return a[1]; }
		__host__ __device__ float z() { return a[2]; }

		__host__ __device__ float& operator[](int c) {
			return a[c];
		}

		__host__ __device__ float operator[](int c) const {
			return a[c];
		}

		__host__ __device__ Vec3 operator-() {
			return Vec3(-a[0], -a[1], -a[2]);
		}

		__host__ __device__ Vec3& operator+=(const Vec3& b) {
			for (int i = 0; i < 3; i++) a[i] += b[i];
			return *this;
		}

		__host__ __device__ Vec3& operator*=(const Vec3& b) {
			for (int i = 0; i < 3; i++) a[i] *= b[i];
			return *this;
		}

		__host__ __device__ float len() const {
			return sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
		}

		__host__ __device__ Vec3 operator+(const Vec3& v) const {
			return Vec3(a[0] + v[0], a[1] + v[1], a[2] + v[2]);
		}

		__host__ __device__ Vec3 operator-(const Vec3& v) const {
			return Vec3(a[0] - v[0], a[1] - v[1], a[2] - v[2]);
		}

		__host__ __device__ Vec3 operator/(float k) const {
			return Vec3(a[0] / k, a[1] / k, a[2] / k);
		}

		__host__ __device__ Vec3 operator*(float k) const {
			return Vec3(a[0] * k, a[1] * k, a[2] * k);
		}

		__host__ __device__ Vec3 operator*(const Vec3& v) const {
			return Vec3(a[0] * v[0], a[1] * v[1], a[2] * v[2]);
		}

		__host__ __device__ bool operator!=(const Vec3& v) const {
			return a[0] == v[0] && a[1] == v[1] && a[2] == v[2];
		}


	private:
		float a[3];
	};

	__host__ __device__ Vec3 operator-(const Vec3& u);
	__host__ __device__ float dot(const Vec3& u, const Vec3& v);
	__host__ __device__ Vec3 reflect(const Vec3& l, const Vec3& n);
	__host__ __device__ Vec3 cross(const Vec3& u, const Vec3& v);
	__host__ __device__ Vec3 norm(const Vec3& u);
	__host__ __device__ float det(const Vec3& c1, const Vec3& c2, const Vec3& c3);
	__host__ __device__ Vec3 clamp(const Vec3& u, const Vec3& min, const Vec3& max);
	__host__ __device__ Vec3 clamp_color(const Vec3& u);
	__host__ __device__ abgr_t vec3_to_abgr(const Vec3& u);
	__host__ std::ostream& operator<<(std::ostream& os, const Vec3& vec);

}

#endif // !VEC3_H