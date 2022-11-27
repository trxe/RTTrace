#define _USE_MATH_DEFINES

#include <cmath>

#include "glm/glm.hpp"
#include "Camera.h"
#include "Ray.h"

namespace RTTrace {
	void Camera::perspective(float fovy, Vec3 focal_point, float res_x, float res_y, float near_plane, float far_plane) {
		// the focal plane is the foot of the perpendicular
		auto aspect = res_x / res_y;
		auto fovy_rad = fovy * M_PI / 180;

		focal_dist = -(focal_point - eye).z;
		this->focal_point = focal_point;
		float image_height = focal_dist * tan(fovy_rad / 2.0) * 2.0;
		float image_width = aspect * image_height;
		pixel_width = image_width / res_x;
		pixel_height = image_height / res_y;

		image_origin_cam = Vec3(-image_width * 0.5f, -image_height * 0.5f, focal_dist);
		this->focal_point = focal_point;
		this->res_x = res_x;
		this->res_y = res_y;

	}

	Ray Camera::gen_ray(int px, int py) {
		auto target_cam = image_origin_cam + Vec3(((float)px + 0.5) * pixel_width, ((float)py + 0.5) * pixel_height, 0);
		auto target_world = target_cam.x * x + target_cam.y * y + target_cam.z * z; // no origin
		return Ray(eye, target_world);
	}
}
