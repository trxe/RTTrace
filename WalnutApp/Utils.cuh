#ifndef UTILS_H 
#define UTILS_H 
#include <stdint.h>
#include "cuda_runtime.h"

#include "Vec3.cuh"

using abgr_t = uint32_t;

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line);

namespace RTTrace {
	/**
	 * Camera settings set from Main.cpp and sent via the Renderer to device.
	 */
	struct CameraInfo {
		float eye[3] = { 0, 0, 6 };
		float at[3] = { 0, 0, 0 };
		float up[3] = { 0, 1, 0 };
		float fovy = 45; // in degrees
		float focal_dist = 5; // in camera space
	};

	/**
	 * Mortion bit shifting pre-req for generating morton codes.
	 */
	__host__ __device__ uint32_t left_shift_3(uint32_t x);

	/**
	 * Generates morton code required for LBVH generation.
	 * 
	 * \param pos World space coordinates of centroid of object.
	 * \param global_min World space coordinates of the minimum point on the global AABB.
	 * \param global_max World space coordinates of the maximum point on the global AABB.
	 */
	__host__ __device__ uint32_t generate_morton_code(const Vec3& pos, const Vec3& global_min, const Vec3& global_max);
}
#endif
