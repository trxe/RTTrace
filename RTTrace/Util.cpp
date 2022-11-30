
#include <iostream>

#include "cuda_runtime.h"
#include "glm/glm.hpp"

#include "Vec3.cuh"

using Vec4 = glm::vec4;
using abgr_t = uint32_t;

namespace RTTrace {
	void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
		if (result) {
			std::cerr << "CUDA error: " << static_cast<unsigned int>(result) << " => " << cudaGetErrorString(result) << "\nat " <<
			file << ":" << line << " '" << func << "' \n";
			// Make sure we call CUDA Device Reset before exiting
			cudaDeviceReset();
			exit(99);
		}
	}

	uint32_t ColorToARGB(const Vec4& u) {
		abgr_t result;
		Vec4 unorm = glm::normalize(u);
		int r = (int)(fabs(unorm[0]) * 256.0);
		int g = (int)(fabs(unorm[1]) * 256.0);
		int b = (int)(fabs(unorm[2]) * 256.0);
		int a = (int)(fabs(unorm[3]) * 256.0);
		result = a << 24 | b << 16 | g << 8 | r;
		return result;
	}

	uint32_t Vec3ToARGB(const Vec3 &u) {
		abgr_t result;
		Vec3 unorm = norm(u);
		int r = (int)(fabs(unorm[0]) * 256.0);
		int g = (int)(fabs(unorm[1]) * 256.0);
		int b = (int)(fabs(unorm[2]) * 256.0);
		result = 0xff000000 | b << 16 | g << 8 | r;
		return result;
	}
}

