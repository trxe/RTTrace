#include <iostream>

#include "cuda_runtime.h"

#include "Camera.cuh"
#include "Ray.cuh"
#include "Renderer.cuh"
#include "Surface.cuh"
#include "Utils.cuh"

#define BLOCK_LENGTH 32

namespace RTTrace {

	__device__ abgr_t get_light(const HitInfo& hit, const LightInfo* lights, int light_count, const SurfaceInfo* surfaces, int surface_count) {
		if (hit.surface_index < 0) return 0x0;
		const SurfaceInfo& surface = surfaces[hit.surface_index];
		Vec3 color;
		for (int i = 0; i < light_count; i++) {
			const LightInfo& l = lights[i];
			Ray shadow_ray{hit.pos, l.origin - hit.pos};
			HitInfo shadow_hit;
			for (int s = 0; s < surface_count; s++) {
				// ignore this surface.
				if (i == hit.surface_index) continue;

				switch (surfaces[i].type) {
				case SurfaceInfo::PLANE:
					hit_plane(shadow_ray, surfaces[i], shadow_hit);
					break;
				case SurfaceInfo::SPHERE:
					hit_sphere(shadow_ray, surfaces[i], shadow_hit);
					break;
				}
			}
			if (!shadow_hit.is_hit) {
				float n_dot_l = dot(hit.norm, shadow_ray.dir);
				color += l.color * surface.mat.kd * l.intensity * fmaxf(0.0, n_dot_l);
			}
		}
		color += surface.mat.ka;
		abgr_t result = vec3_to_abgr(color);
		return result;
	}

	// __global__ void gpu_render(const CameraInfo* info, int width, int height, abgr_t* data, const SurfaceInfo* surfaces, int surface_count) {
	__global__ void gpu_render(const CameraInfo* info, int width, int height, abgr_t* data, const SurfaceInfo* surfaces, int surface_count, const LightInfo* lights, int light_count) {
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		// need to invert position
		int pos = (height - y) * width + (width - x);
		if (x >= width || y >= height) return;
		Camera c(*info, width, height);
		Ray ray = c.gen_ray(x, y);
		HitInfo hit_global;

		for (int i = 0; i < surface_count; i++) {
			HitInfo hit;
			switch (surfaces[i].type) {
			case SurfaceInfo::PLANE:
				hit_plane(ray, surfaces[i], hit);
				break;
			case SurfaceInfo::SPHERE:
				hit_sphere(ray, surfaces[i], hit);
				break;
			}
			if (hit.t < hit_global.t) {
				hit_global = hit;
				hit_global.surface_index = i;
			}
		}
		// data[pos] = hit_global.is_hit ? vec3_to_abgr(norm(hit_global.norm)) : 0x0;
		data[pos] = hit_global.is_hit ? get_light(hit_global, lights, light_count, surfaces, surface_count) : 0x0;
		// data[pos] = hit_global.is_hit ? 0xff5f0000 : 0x0;
	}

	void Renderer::set_world(SurfaceInfo* surfaces, int count) {
		surface_count = count;
		if (surfaces_d != nullptr) {
			checkCudaErrors(cudaFree(&surfaces_d));
		}
		checkCudaErrors(cudaMalloc(&surfaces_d, sizeof(SurfaceInfo) * count));
		checkCudaErrors(cudaMemcpy(surfaces_d, surfaces, sizeof(SurfaceInfo) * count, cudaMemcpyHostToDevice));
	}

	void Renderer::set_lights(LightInfo* lights, int count) {
		light_count = count;
		if (lights_d != nullptr) {
			checkCudaErrors(cudaFree(&lights_d));
		}
		checkCudaErrors(cudaMalloc(&lights_d, sizeof(LightInfo) * count));
		checkCudaErrors(cudaMemcpy(lights_d, lights, sizeof(LightInfo) * count, cudaMemcpyHostToDevice));
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
		gpu_render<<<gridDim,blockDim>>>(info_device, static_cast<int>(viewport_width), static_cast<int>(viewport_height), data_d, surfaces_d, surface_count, lights_d, light_count);
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaMemcpy(data, data_d, pixel_count * sizeof(abgr_t), cudaMemcpyDeviceToHost));
	}
}
