#pragma once

#define _USE_MATH_DEFINES

#include <math.h>

#include "glm/glm.hpp"
#include "Ray.h"

using Vec3 = glm::vec3;

namespace RTTrace {
	class Camera {
	public:
		Camera() = delete;

		Camera(const Vec3& eye, const Vec3& at = { 0, 0, 0 }, const Vec3& up = { 0, 1, 0 }, int rays_per_pixel = { 1 }) :
			eye{ eye }, at{ at }, up{ up }, rays_per_pixel{ rays_per_pixel } {
			z = glm::normalize(eye - at);
			x = glm::normalize(glm::cross(up, z));
			y = glm::normalize(glm::cross(z, x));
			// default perspective
			perspective(45, at, 1, 1);
		};

		void perspective(float fovy, Vec3 focal_point, float res_x, float res_y, float near_plane = { 0.1f }, float far_plane = { 100 });
		Ray gen_ray(int px, int py);

	private:
		Vec3 eye, at, up;
		Vec3 x, y, z; // camera space axes
		Vec3 image_origin_cam; // in camera coordinates
		Vec3 focal_point;
		glm::mat4 world_to_cam;
		glm::mat4 cam_to_clip;
		float focal_dist;
		float pixel_width, pixel_height;
		float res_x, res_y;
		int rays_per_pixel;
	};
}
