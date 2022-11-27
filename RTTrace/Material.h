
#include "Ray.h"
#include "glm/glm.hpp"

using Color = glm::vec4;

namespace RTTrace {

	class Material {
	public:
		// Returns true if scattered ray is created.
		virtual bool render(const Ray& r, Ray& scattered, Color& color) = 0;
	protected:
		Color ka, kd, ks;

	};
}