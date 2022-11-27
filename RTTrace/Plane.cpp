#include <memory>
#include <vector>

#include "glm/glm.hpp"

#include "Ray.h"
#include "Surface.h"

namespace RTTrace {
	bool Plane::hit(const Ray& ray, HitInfo& info) const {
		auto d = glm::normalize(ray.dir);
		float n_dot_dir = glm::dot(normal, d);
		if (n_dot_dir == 0) return false; // ray parallel to plane
		float n_dot_ori = glm::dot(normal, ray.origin);
		auto t = (n_dot_point - n_dot_ori) / n_dot_dir;
		if (t < T_EPSILON || t > T_MAX) return false;
		info.t = t;
		info.norm = normal;
		info.pos = ray.point(t);
		return true;
	}
}
