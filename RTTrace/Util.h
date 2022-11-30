#pragma once

#include "cuda_runtime.h"
#include "glm/glm.hpp"

#include "Vec3.cuh"

#ifndef UTIL_H
#define UTIL_H

using Vec4 = glm::vec4;
using abgr_t = uint32_t;
using std::fabs;

namespace RTTrace {
	#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
	const float T_EPSILON = 0.0001f;
	const float T_MAX = 10000;

	uint32_t ColorToARGB(const Vec4& u);

	uint32_t Vec3ToARGB(const Vec3& u);

	void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line);

}

#endif
