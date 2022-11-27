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

	uint32_t ColorToARGB(const Vec4& u);

	uint32_t Vec3ToARGB(const Vec3& u);

	std::ostream& operator<<(std::ostream& os, const Vec3& vec);

}
