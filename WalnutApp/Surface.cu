#include "Surface.cuh"
#include "Vec3.cuh"

namespace RTTrace {
	__device__ bool hit_sphere (const Ray& ray, const SurfaceInfo& surface, HitInfo& hit) {
		Vec3 origin = surface.origin;
		float radius = surface.scale;

		Vec3 rd = norm(ray.dir);
		Vec3 ro = origin - ray.origin;
		float a = 1.0;
		float b = 2.0 * dot(rd, ro);
		float c = dot(ro, ro) - radius*radius;

		float discSquared = b * b - 4 * a * c;
		if (discSquared < 0) return false;
		float t_small = (-b - sqrt(discSquared)) / (2 * a);
		float t_big = (-b + sqrt(discSquared)) / (2 * a);
		if (t_small > T_MAX || t_big < T_EPSILON) return false;
		float t = t_small < T_EPSILON ? t_big : t_small;
		hit.is_hit = true;
		hit.t = t;
		hit.pos = ray.point(t);
		hit.norm = norm(hit.pos - origin);
		return true;
	}
	__device__ bool hit_plane (const Ray& ray, const SurfaceInfo& surface, HitInfo& hit) {
		Vec3 normal = surface.normal;
		auto d = norm(ray.dir);
		float n_dot_dir = dot(normal, d);
		if (n_dot_dir == 0) return false; // ray parallel to plane
		float n_dot_ori = dot(normal, ray.origin);
		float n_dot_point = dot(normal, surface.origin);
		auto t = (n_dot_point - n_dot_ori) / n_dot_dir;
		if (t < T_EPSILON || t > T_MAX) return false;
		hit.is_hit = true;
		hit.t = t;
		hit.pos = ray.point(t);
		hit.norm = normal;
		return true;	
	}

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