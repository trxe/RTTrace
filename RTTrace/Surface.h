#pragma once

#include <memory>
#include <vector>

#include "glm/glm.hpp"

#include "Ray.cuh"

using std::shared_ptr;
using std::make_shared;
using Vec4 = glm::vec4;

namespace RTTrace {
	struct HitInfo {
		HitInfo() : t{ -1 }, pos{ 0,0,0 }, norm{ 0,0,0 } {};
		float t;
		Vec3 pos;
		Vec3 norm;
		// Material
	};

	class Surface {
	public:
		virtual bool hit(const Ray& r, HitInfo& info) const = 0;
	protected:
		Vec3 origin{ 0,0,0 };
	};

	class Plane : public Surface {
	public:
		Plane() = delete;
		Plane(const Vec3& origin, const Vec3& normal) : normal{ normal }, n_dot_point{ -dot(normal,origin) } {
			Surface::origin = origin;
		}
		virtual bool hit(const Ray& ray, HitInfo& info) const override;
	private:
		Vec3 normal;
		float n_dot_point;
	};

	class Sphere : public Surface {
	public:
		Sphere() = delete;
		Sphere(const Vec3& origin, float radius) : radius{ radius } {
			Surface::origin = origin;
		};
		virtual bool hit(const Ray& ray, HitInfo& info) const override;
	private:
		float radius;
	};


}
