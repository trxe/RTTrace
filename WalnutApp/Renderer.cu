#include <iostream>

#include "cuda_runtime.h"

#include "Camera.cuh"
#include "Ray.cuh"
#include "Renderer.cuh"
#include "Surface.cuh"
#include "Utils.cuh"

#define BLOCK_LENGTH 32

namespace RTTrace {

	__global__ void gpu_render(const CameraInfo* info, int width, int height, abgr_t* data) {
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		int pos = y * width + x;
		if (x >= width || y >= height) return;
		Camera c(*info, width, height);
		Ray ray = c.gen_ray(x, y);
		data[pos] = vec3_to_abgr(ray.origin + ray.dir);
	}

	/*
	void Renderer::set_world(Surface** surfaces, int count) {
		surface_count = count;
		checkCudaErrors(cudaFree(&surfaces_d));
		checkCudaErrors(cudaMalloc(&surfaces_d, sizeof(Surface) * count));
		for (int i = 0; i < count; i++) {
			checkCudaErrors(cudaMemcpy(&surfaces_d[i], &surfaces[i], sizeof(Surface), cudaMemcpyHostToDevice));
		}
	}
	*/

	void BasicRaytracer::render(float viewport_width, float viewport_height, const CameraInfo& info, abgr_t* data) {
		size_t pixel_count = static_cast<size_t>(viewport_width * viewport_height);
		// this condition literally saves 1ms HAHAH
		if (last_res[0] != viewport_width || last_res[1] != viewport_height) {
			last_res[0] = viewport_width;
			last_res[1] = viewport_height;
			checkCudaErrors(cudaFree(data_d));
			checkCudaErrors(cudaMalloc(&data_d, pixel_count * sizeof(abgr_t)));
		}

		dim3 gridDim(ceil(viewport_width / BLOCK_LENGTH), ceil(viewport_height / BLOCK_LENGTH));
		dim3 blockDim(BLOCK_LENGTH, BLOCK_LENGTH);

		CameraInfo* info_device;
		checkCudaErrors(cudaMalloc(&info_device, sizeof(CameraInfo)));
		checkCudaErrors(cudaMemcpy(info_device, &info, sizeof(CameraInfo), cudaMemcpyHostToDevice));

		checkCudaErrors(cudaDeviceSynchronize());
		gpu_render<<<gridDim,blockDim>>>(info_device, static_cast<int>(viewport_width), static_cast<int>(viewport_height), data_d);
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaMemcpy(data, data_d, pixel_count * sizeof(abgr_t), cudaMemcpyDeviceToHost));
	}
}
