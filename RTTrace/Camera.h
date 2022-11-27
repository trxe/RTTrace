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

		void perspective(float fovy, Vec3 focal_point, float res_x, float res_y, float near_plane = { 0.1f }, float far_plane = { 100 }) {
			// the focal plane is the foot of the perpendicular
			auto aspect = res_x / res_y;
			auto fovy_rad = fovy * M_PI / 180;

			focal_dist = -(focal_point - eye).z;
			this->focal_point = focal_point;
			auto image_height = focal_dist * tan(fovy_rad / 2.0) * 2.0;
			auto image_width = aspect * image_height;
			pixel_width = image_width / res_x;
			pixel_height = image_height / res_y;

			image_origin_cam = Vec3(-image_width * 0.5f, -image_height * 0.5f, focal_dist);
			this->focal_point = focal_point;
			this->res_x = res_x;
			this->res_y = res_y;

		}

		Ray gen_ray(int px, int py) {
			auto target_cam = image_origin_cam + Vec3(((float)px + 0.5) * pixel_width, ((float)py + 0.5) * pixel_height, 0);
			auto target_world = target_cam.x * x + target_cam.y * y + target_cam.z * z; // no origin
			return Ray(eye, target_world);
		}

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
