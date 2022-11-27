
#include <iostream>

#include <glm/glm.hpp>
#include "Ray.h"

namespace RTTrace {
	std::ostream& operator<<(std::ostream& os, const Ray& ray) {
		os << "origin: {" << ray.origin << "}, dir: {" << ray.dir << "}";
		return os;
	}
}