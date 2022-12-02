#ifndef UTILS_H 
#define UTILS_H 
#include <stdint.h>
#include "cuda_runtime.h"

using abgr_t = uint32_t;

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line);

namespace RTTrace {
	/**
	 * Camera settings set from Main.cpp and sent via the Renderer to device.
	 */
	struct CameraInfo {
		float eye[3] = { 0, 0, 1 };
		float at[3] = { 0, 0, 0 };
		float up[3] = { 0, 1, 0 };
		float fovy = 45; // in degrees
		float focal_dist = 5; // in camera space
	};
}
#endif
