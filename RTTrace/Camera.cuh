#pragma once

#define _USE_MATH_DEFINES

#include <math.h>

#include "cuda_runtime.h"
#include "glm/glm.hpp"

#include "Ray.cuh"
#include "Vec3.cuh"

#ifndef CAMERA_H
#define CAMERA_H


namespace RTTrace {
	extern __constant__ float px_width;
	extern __constant__ float px_height;
	/*
	extern __constant__ float cam_x[3];
	extern __constant__ float cam_y[3];
	extern __constant__ float cam_z[3];
	extern __constant__ float image_origin_cam_d[3];
	*/

	class Camera {
	public:
		Camera() = delete;

		__device__ Camera(const Vec3& eye, const Vec3& at = { 0, 0, 0 }, const Vec3& up = { 0, 1, 0 }, int rays_per_pixel = { 1 }) :
			eye{ eye }, at{ at }, up{ up }, rays_per_pixel{ rays_per_pixel } {
			z = norm(eye - at);
			x = norm(cross(up, z));
			y = norm(cross(z, x));
			// default perspective
			perspective(45, at, 1, 1);
		};

		__device__ void perspective(float fovy, Vec3 focal_point, float res_x, float res_y, float near_plane = { 0.1f }, float far_plane = { 100 });
		__device__ Ray gen_ray(int px, int py);
		__device__ int get_res_x() { return (int)res_x; };
		__device__ int get_res_y() { return (int)res_y; };

	private:
		Vec3 eye, at, up;
		Vec3 x, y, z; // camera space axes
		Vec3 image_origin_cam; // in camera coordinates
		Vec3 focal_point;
		float focal_dist;
		float pixel_width, pixel_height;
		float res_x, res_y;
		int rays_per_pixel;
	};
}
#endif