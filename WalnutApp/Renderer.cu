#include <iostream>

#include "cuda_runtime.h"

#include "Camera.cuh"
#include "Ray.cuh"
#include "Renderer.cuh"
#include "Surface.cuh"
#include "Utils.cuh"

#define BLOCK_LENGTH 32

namespace RTTrace {

	// __global__ void gpu_render(const CameraInfo* info, int width, int height, abgr_t* data) {
	__global__ void gpu_render(const CameraInfo* info, int width, int height, abgr_t* data, const SurfaceInfo* surfaces, int surface_count) {
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		int pos = y * width + x;
		if (x >= width || y >= height) return;
		Camera c(*info, width, height);
		Ray ray = c.gen_ray(x, y);
		HitInfo hit;

		for (int i = 0; i < surface_count; i++) {
			switch (surfaces[i].type) {
			case SurfaceInfo::PLANE:
				hit_plane(ray, surfaces[i], hit);
				break;
			case SurfaceInfo::SPHERE:
				hit_sphere(ray, surfaces[i], hit);
				break;
			}
		}
		data[pos] = hit.is_hit ? vec3_to_abgr(ray.origin + ray.dir) : 0x0;
	}

	void Renderer::set_world(SurfaceInfo* surfaces, int count) {
		surface_count = count;
		if (surfaces_d != nullptr) {
			checkCudaErrors(cudaFree(&surfaces_d));
		}
		checkCudaErrors(cudaMalloc(&surfaces_d, sizeof(SurfaceInfo) * count));
		checkCudaErrors(cudaMemcpy(surfaces_d, surfaces, sizeof(SurfaceInfo) * count, cudaMemcpyHostToDevice));
	}

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
		// gpu_render<<<gridDim,blockDim>>>(info_device, static_cast<int>(viewport_width), static_cast<int>(viewport_height), data_d);
		gpu_render<<<gridDim,blockDim>>>(info_device, static_cast<int>(viewport_width), static_cast<int>(viewport_height), data_d, surfaces_d, surface_count);
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaMemcpy(data, data_d, pixel_count * sizeof(abgr_t), cudaMemcpyDeviceToHost));
	}
}
