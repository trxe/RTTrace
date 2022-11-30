
#include <iostream>

#include "Ray.cuh"

namespace RTTrace {
	std::ostream& operator<<(std::ostream& os, const Ray& ray) {
		os << "origin: {" << ray.origin << "}, dir: {" << ray.dir << "}";
		return os;
	}
}