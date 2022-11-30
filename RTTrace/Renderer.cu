#include "cuda_runtime.h"
#include "Camera.cuh"
#include "Renderer.cuh"
#include "Util.h"

using abgr_t = uint32_t;


namespace RTTrace {
	__constant__ float eye[3];
	__constant__ float at[3];
	__constant__ float up[3];

	__device__ void mult(const float* a, const float* b, float* out, int count) {
		for (int i = 0; i < count; i++) {
			out[i] = a[i] * b[i];
		}
	}

	__device__ void add(const float* a, const float* b, float* out, int count) {
		for (int i = 0; i < count; i++) {
			out[i] = a[i] + b[i];
		}
	}

	/*
	__device__ abgr_t float3ToARGB(const float* u) {
		abgr_t result;
		float len = sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
		float unorm[3] = { u[0] / len, u[1] / len, u[2] / len };
		int r = (int)(fabs(unorm[0]) * 256.0);
		int g = (int)(fabs(unorm[1]) * 256.0);
		int b = (int)(fabs(unorm[2]) * 256.0);
		result = 0xff000000 | b << 16 | g << 8 | r;
		return result;
	}
	*/

	__device__ abgr_t VecToARGB(const Vec3 &u) {
		abgr_t result;
		Vec3 unorm = norm(u);
		int r = (int)(fabs(unorm[0]) * 256.0);
		int g = (int)(fabs(unorm[1]) * 256.0);
		int b = (int)(fabs(unorm[2]) * 256.0);
		result = 0xff000000 | b << 16 | g << 8 | r;
		return result;
	}

    __global__ void gen_ray(abgr_t* data, int width, int height) {
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		int pos = y * width + x;
		if (x >= width || y >= height) return;
		Vec3 d(3, 2, 1000);
		/*
		Vec3 e = d + d;
		printf("rd: %f %f %f\n", e[0], e[1], e[2]);
		Ray r(d, d);
		*/
		Camera camera;
		Ray r = camera.gen_ray(x, y);
		printf("rd: %f %f %f\n", r.dir[0], r.dir[1], r.dir[2]);
		// data[pos] = VecToARGB(r.dir);

		/*
		float tcam[3] = {
			image_origin_cam_d[0],
			image_origin_cam_d[1] + (float)x + 0.5 * px_width,
			image_origin_cam_d[2] + (float)y + 0.5 * px_height
		};

		float tworld_x[3], tworld_y[3], tworld_z[3];
		mult(tcam, cam_x, tworld_x, 3);
		mult(tcam, cam_y, tworld_y, 3);
		mult(tcam, cam_z, tworld_z, 3);

		float tworld[3] = {0.0, 0.0, 0.0};
		add(tworld, tworld_x, tworld, 3);
		add(tworld, tworld_y, tworld, 3);
		add(tworld, tworld_z, tworld, 3);

		data[pos] = float3ToARGB(tworld);
		*/
    }

    abgr_t* const Renderer::render(float viewport_width, float viewport_height) {
		size_t pixel_count = static_cast<size_t>(viewport_width * viewport_height);
		delete[] data;
		data = new abgr_t[pixel_count];
		abgr_t* data_d = new abgr_t[pixel_count];
		checkCudaErrors(cudaMalloc(&data_d, pixel_count * sizeof(abgr_t)));

        dim3 blockSize(32, 32);
        dim3 gridSize(ceil(viewport_width / 32), ceil(viewport_height / 32));

		gen_ray<<<gridSize, blockSize>>>(data_d, static_cast<int>(viewport_width), static_cast<int>(viewport_height);
		checkCudaErrors(cudaMemcpy(data, data_d, pixel_count * sizeof(abgr_t), cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaDeviceSynchronize());

        // Serial implementation
		/*
		for (int y = 0; y < viewport_height; y++) {
			for (int x = 0; x < viewport_width; x++) {
				Ray r = camera.gen_ray(x, y);
				int index = (int)y * viewport_width + (int)x;
				HitInfo hit{};
				for (auto &surface : surfaces) {
					HitInfo this_hit;
					bool is_hit = surface->hit(r, this_hit);
					if (is_hit) {
						hit = (hit.t < 0 || this_hit.t < hit.t) ? this_hit : hit;
					}
				}
				data[index] = hit.t > 0 ? Vec3ToARGB(glm::normalize(r.dir)) : 0x0;
			}
		}
		*/

        return data;
    }
}
