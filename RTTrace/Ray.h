#pragma once

#include <iostream>

#include <glm/glm.hpp>

#include "Util.h"

using Vec3 = glm::vec3;

namespace RTTrace {
	struct Ray {
		Ray() = delete;
		Ray(Vec3 origin, Vec3 dir) : origin{ origin }, dir{ glm::normalize(dir)} {};

		Vec3 origin;
		Vec3 dir;

		inline Vec3 point(float t) const {
			return origin + t * dir;
		}

		friend std::ostream& operator<<(std::ostream& os, const Ray& ray);
	};

	std::ostream& operator<<(std::ostream& os, const Ray& ray) {
		os << "origin: {" << ray.origin << "}, dir: {" << ray.dir << "}";
		return os;
	}
}

