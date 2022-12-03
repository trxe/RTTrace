#ifndef LIGHT_H
#define LIGHT_H

#include "Vec3.cuh"

namespace RTTrace {
	struct LightInfo {
		enum Type { POINT };
		Type type;
		Vec3 origin;
		Vec3 color;
		float intensity;
	};
}

#endif // !LIGHT_H
