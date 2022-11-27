
#include <memory>
#include <vector>

#include "glm/glm.hpp"

#include "Ray.h"
#include "Surface.h"

namespace RTTrace {
	bool Sphere::hit(const Ray& ray, HitInfo& info) const {
		auto rd = glm::normalize(ray.dir);
		auto ro = origin - ray.origin;
		auto a = 1;
		auto b = 2 * glm::dot(rd, ro);
		auto c = glm::dot(ro, ro) - radius*radius;

		auto discSquared = b * b - 4 * a * c;
		if (discSquared < 0) return false;
		auto t_small = (-b - std::sqrt(discSquared)) / (2 * a);
		auto t_big = (-b + std::sqrt(discSquared)) / (2 * a);
		if (t_small > T_MAX || t_big < T_EPSILON) return false;
		float t = t_small < T_EPSILON ? t_big : t_small;
		info.t = t;
		info.pos = ray.point(t);
		info.norm = glm::normalize(info.pos - origin);
		return true;
	}
}
