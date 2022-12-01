#include "cuda.h"
#include "cuda_runtime.h"

#include "Renderer.cuh"

namespace RTTrace {
	// __constant__ float help;

	__global__ void kernel(int width, int height) {
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		int pos = y * width + x;
		if (x >= width || y >= height) return;
		if (!x && !y) printf("babla");
	}

	void BasicRaytracer::render(float viewport_width, float viewport_height, const CameraInfo& info, abgr_t* data) {
		kernel <<<1,1>>> (static_cast<int>(viewport_width), static_cast<int>(viewport_height));
	}
}
