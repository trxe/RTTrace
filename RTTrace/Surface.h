#pragma once

#include <memory>
#include <vector>

#include "glm/glm.hpp"

#include "Ray.h"

using std::shared_ptr;
using std::make_shared;
using Vec3 = glm::vec3;
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
		Plane(const Vec3& origin, const Vec3& normal) : normal{ normal }, n_dot_point{ -glm::dot(normal,origin) } {
			Surface::origin = origin;
		}

		virtual bool hit(const Ray& ray, HitInfo& info) const override {
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
		virtual bool hit(const Ray& ray, HitInfo& info) const override {
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
	private:
		float radius;
	};

	std::vector<shared_ptr<Surface>> generate_default_surfaces() {
		std::vector<shared_ptr<Surface>> surface_list;
		surface_list.emplace_back(make_shared<Sphere>(Vec3(0.0,0.0,0.0), 1.2));
		surface_list.emplace_back(make_shared<Plane>(Vec3(0.0,-2.0, 0.0), Vec3(0.0, 1.0, 0.0)));
		return surface_list;
	}

}
