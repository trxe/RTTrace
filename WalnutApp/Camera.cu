#define _USE_MATH_DEFINES

#include <cmath>

#include "cuda_runtime.h"
#include "glm/glm.hpp"
#include "Camera.cuh"
#include "Utils.cuh"
#include "Ray.cuh"

namespace RTTrace {

	__device__ Ray Camera::gen_ray(int px, int py) {
		Vec3 target_cam = norm(image_origin_cam + Vec3(((float)px + 0.5) * pixel_width, ((float)py + 0.5) * pixel_height, 0));
		Vec3 target_world = x * target_cam[0] + y * target_cam[1] + z * target_cam[2]; // no origin
		if (isnan(target_world[0])) { 
			printf("error: ray %d %d\n", x, y);
		}
		return Ray(eye, target_world);
	}
}