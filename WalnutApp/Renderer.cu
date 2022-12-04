#include <iostream>

#include "cuda_runtime.h"

#include "Camera.cuh"
#include "Ray.cuh"
#include "Renderer.cuh"
#include "Surface.cuh"
#include "Utils.cuh"

#define BLOCK_LENGTH 32

namespace RTTrace {

	__device__ Vec3 get_light(HitInfo& hit, const LightInfo* lights, int light_count, const SurfaceInfo* surfaces, int surface_count) {
		if (hit.surface_index < 0) return 0x0;
		const SurfaceInfo& surface = surfaces[hit.surface_index];
		Vec3 color;
		for (int i = 0; i < light_count; i++) {
			const LightInfo& l = lights[i];
			Ray shadow_ray{hit.pos, l.origin - hit.pos};
			HitInfo shadow_hit;
			for (int s = 0; s < surface_count && !shadow_hit.is_hit; s++) {
				// ignore this surface.
				if (s == hit.surface_index) continue;

				switch (surfaces[s].type) {
				case SurfaceInfo::PLANE:
					hit_plane(shadow_ray, surfaces[s], shadow_hit);
					break;
				case SurfaceInfo::SPHERE:
					hit_sphere(shadow_ray, surfaces[s], shadow_hit);
					break;
				case SurfaceInfo::TRIANGLE:
					hit_triangle(shadow_ray, surfaces[s], shadow_hit);
					break;
				}
			}
			if (!shadow_hit.is_hit) {
				float n_dot_l = dot(hit.norm, shadow_ray.dir);
				if (n_dot_l <= 0.0) continue;
				color += l.color * surface.mat.kd * l.intensity * n_dot_l;
				Vec3 r = reflect(shadow_ray.dir, hit.norm);
				hit.refl_dir = r;
				float r_dot_v = fmaxf(0, dot(r, norm(-hit.view_dir)));
				float r_dot_v_pow_n = powf(r_dot_v, surface.mat.n);
				Vec3 specular = l.color * surface.mat.ks * l.intensity * r_dot_v_pow_n;
				color += specular;
			}
		}
		color += surface.mat.ka;
		return color;
	}

	__global__ void gpu_render(
			const CameraInfo* info, int width, int height, abgr_t* data, 
			const SurfaceInfo* surfaces, int surface_count, 
			const LightInfo* lights, int light_count, int recursion_levels) {
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		// need to invert position
		int pos = (height - y) * width + (width - x);
		if (x >= width || y >= height) return;
		Camera c(*info, width, height);
		Ray ray = c.gen_ray(x, y);
		Vec3 color{};
		Vec3 att = Vec3(1, 1, 1);
		for (int r = 0; r <= recursion_levels; r++) {
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
				case SurfaceInfo::TRIANGLE:
					hit_triangle(ray, surfaces[i], hit);
					break;
				}
				if (hit.t < hit_global.t) {
					hit_global = hit;
					hit_global.surface_index = i;
				}
			}

			if (hit_global.is_hit) {
				color += get_light(hit_global, lights, light_count, surfaces, surface_count) * att;
				// set up next level of recursion: light and reflection ray
				att *= surfaces[hit_global.surface_index].mat.krg;
				ray.origin = hit_global.pos;
				ray.dir = norm(hit_global.refl_dir);
			}
		}
		data[pos] = vec3_to_abgr(color);
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

	void BasicRaytracer::render(float viewport_width, float viewport_height, int recursion_levels, const CameraInfo& info, abgr_t* data) {
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
		gpu_render<<<gridDim,blockDim>>>(info_device, static_cast<int>(viewport_width), static_cast<int>(viewport_height), data_d, surfaces_d, surface_count, lights_d, light_count, recursion_levels);
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaMemcpy(data, data_d, pixel_count * sizeof(abgr_t), cudaMemcpyDeviceToHost));
	}
}
