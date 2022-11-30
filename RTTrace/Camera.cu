#define _USE_MATH_DEFINES

#include <cmath>

#include "cuda_runtime.h"
#include "glm/glm.hpp"
#include "Camera.cuh"
#include "Ray.cuh"

namespace RTTrace {

	/*
	__constant__ float cam_x[3];
	__constant__ float cam_y[3];
	__constant__ float cam_z[3];
	__constant__ float image_origin_cam_d[3];
	*/
	__constant__ float px_width;
	__constant__ float px_height;

	__host__ __device__ void Camera::perspective(float fovy, Vec3 focal_point, float res_x, float res_y, float near_plane, float far_plane) {
		// the focal plane is the foot of the perpendicular
		auto aspect = res_x / res_y;
		auto fovy_rad = fovy * M_PI / 180;

		focal_dist = -(focal_point - eye).z();
		this->focal_point = focal_point;
		float image_height = focal_dist * tan(fovy_rad / 2.0) * 2.0;
		float image_width = aspect * image_height;
		pixel_width = image_width / res_x;
		pixel_height = image_height / res_y;

		image_origin_cam = Vec3(-image_width * 0.5f, -image_height * 0.5f, focal_dist);
		checkCudaErrors(cudaMemcpyToSymbol(px_width, &pixel_width, sizeof(float)));
		checkCudaErrors(cudaMemcpyToSymbol(px_height, &pixel_height, sizeof(float)));
		/*
		float cam_x_host[3] = { x.x(), x.y(), x.z()};
		float cam_y_host[3] = { y.x(), y.y(), y.z() };
		float cam_z_host[3] = { z.x(), z.y(), z.z() };
		float image_origin_cam_host[3] = { image_origin_cam.x(), image_origin_cam.y(), image_origin_cam.z() };
		checkCudaErrors(cudaMemcpyToSymbol(cam_x, cam_x_host, 3 * sizeof(float)));
		checkCudaErrors(cudaMemcpyToSymbol(cam_y, cam_y_host, 3 * sizeof(float)));
		checkCudaErrors(cudaMemcpyToSymbol(cam_z, cam_z_host, 3 * sizeof(float)));
		checkCudaErrors(cudaMemcpyToSymbol(image_origin_cam_d, image_origin_cam_host, 3 * sizeof(float)));
		*/

		this->focal_point = focal_point;
		this->res_x = res_x;
		this->res_y = res_y;

	}

	__device__ Ray Camera::gen_ray(int px, int py) {
		/*
		Vec3 test(12, 1, 2);
		Vec3 test2 = test + test;
		printf("%f %f %f\n", test2[0], test2[1], test2[2]);
		*/
		printf("%f\n", image_origin_cam[0]);
		__syncthreads();
		Vec3 target_cam = image_origin_cam + Vec3(((float)px + 0.5) * px_width, ((float)py + 0.5) * px_height, 0);
		Vec3 target_world = x * target_cam.x() + y * target_cam.y() + z * target_cam.z(); // no origin
		return Ray(eye, target_world);
	}

}