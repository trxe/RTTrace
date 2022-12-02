#include "Surface.cuh"
#include "Vec3.cuh"

namespace RTTrace {
	/*
	__device__ bool Sphere::hit(const Ray& ray, HitInfo* hit) const {
		auto rd = norm(ray.dir);
		auto ro = origin - ray.origin;
		auto a = 1;
		auto b = 2 * dot(rd, ro);
		auto c = dot(ro, ro) - radius*radius;

		auto discSquared = b * b - 4 * a * c;
		if (discSquared < 0) return false;
		auto t_small = (-b - std::sqrt(discSquared)) / (2 * a);
		auto t_big = (-b + std::sqrt(discSquared)) / (2 * a);
		if (t_small > T_MAX || t_big < T_EPSILON) return false;
		float t = t_small < T_EPSILON ? t_big : t_small;
		if (hit != nullptr) {
			hit->t = t;
			hit->pos = ray.point(t);
			hit->norm = norm(hit->pos - origin);
		}
		return true;
	}
	__device__ bool Plane::hit(const Ray& ray, HitInfo* hit) const {
		auto d = norm(ray.dir);
		float n_dot_dir = dot(normal, d);
		if (n_dot_dir == 0) return false; // ray parallel to plane
		float n_dot_ori = dot(normal, ray.origin);
		auto t = (n_dot_point - n_dot_ori) / n_dot_dir;
		if (t < T_EPSILON || t > T_MAX) return false;
		if (hit != nullptr) {
			hit->t = t;
			hit->pos = ray.point(t);
			hit->norm = normal;
		}
		return true;	
	}
	*/

}
