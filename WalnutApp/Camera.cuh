#ifndef CAMERA_H
#define CAMERA_H

#define _USE_MATH_DEFINES

#include <math.h>

#include "cuda_runtime.h"
#include "glm/glm.hpp"

#include "Ray.cuh"
#include "Utils.cuh"
#include "Vec3.cuh"

namespace RTTrace {

	class Camera {
	public:
		Camera() = delete;

		__device__ Camera(const CameraInfo& bound, int res_x, int res_y) : res_x(res_x), res_y(res_y), focal_dist(bound.focal_dist) {
			eye = Vec3(bound.eye);
			at = Vec3(bound.at);
			up = Vec3(bound.up);
			z = norm(eye - at);
			x = norm(cross(up, z));
			y = norm(cross(z, x));

			// default perspective
			aspect = (float)res_x / (float)res_y;
			fovy_rad = bound.fovy * M_PI / 180;
			image_height = focal_dist * tan(fovy_rad / 2.0) * 2.0;
			image_width = aspect * image_height;
			pixel_width = image_width / (float)res_x;
			pixel_height = image_height / (float)res_y;

			image_origin_cam = Vec3(-image_width * 0.5f, -image_height * 0.5f, -focal_dist);
		}

		__device__ void testDevice() {
			printf("Res x, y: %d, %d\n", res_x, res_y);
			printf("Image width, height: %f, %f\n", image_width, image_height);
			printf("Pixel width, height: %f, %f\n", pixel_width, pixel_height);
			printf("Image Origin: %f %f %f\n", image_origin_cam[0], image_origin_cam[1], image_origin_cam[2]);
		}

		__device__ Ray gen_ray(int px, int py);
		__device__ int get_res_x() { return (int)res_x; };
		__device__ int get_res_y() { return (int)res_y; };

	private:
		Vec3 eye, at, up;
		Vec3 x, y, z; // camera space axes
		Vec3 image_origin_cam; // in camera coordinates
		float aspect;
		float fovy_rad;
		float focal_dist;
		float image_width, image_height;
		float pixel_width, pixel_height;
		int res_x, res_y;
		int rays_per_pixel;
	};
}
#endif