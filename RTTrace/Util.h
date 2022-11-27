#pragma once

#include <stdint.h>
#include <glm/glm.hpp>

using Vec3 = glm::vec3;
using Vec4 = glm::vec4;
using abgr_t = uint32_t;
using std::fabs;

namespace RTTrace {
	const float T_EPSILON = 0.0001f;
	const float T_MAX = 10000;

	static uint32_t ColorToARGB(const Vec4 &u) {
		abgr_t result;
		Vec3 unorm = glm::normalize(u);
		int r = (int)(fabs(unorm[0]) * 256.0);
		int g = (int)(fabs(unorm[1]) * 256.0);
		int b = (int)(fabs(unorm[2]) * 256.0);
		int a = (int)(fabs(unorm[3]) * 256.0);
		result = a << 24 | b << 16 | g << 8 | r;
		return result;
	}

	inline uint32_t Vec3ToARGB(const Vec3 &u) {
		abgr_t result;
		Vec3 unorm = glm::normalize(u);
		int r = (int)(fabs(unorm[0]) * 256.0);
		int g = (int)(fabs(unorm[1]) * 256.0);
		int b = (int)(fabs(unorm[2]) * 256.0);
		result = 0xff000000 | b << 16 | g << 8 | r;
		return result;
	}

	std::ostream& operator<<(std::ostream& os, const Vec3& vec) {
		os << "{";
		for (int i = 0; i < 3; i++) {
			os << vec[i] << " ";
		}
		os << "}";
		return os;
	}

}
