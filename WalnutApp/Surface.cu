#include "Mat.cuh"
#include "Surface.cuh"
#include "Vec3.cuh"

namespace RTTrace {
	__device__ bool hit_sphere (const Ray& ray, const SurfaceInfo& surface, HitInfo& hit) {
		Vec3 origin = surface.origin;
		float radius = surface.scale;

		Vec3 rd = norm(ray.dir);
		Vec3 ro = ray.origin - origin;
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
		hit.view_dir = rd;
		hit.pos = ray.point(t); // dir must be unit vector
		hit.norm = norm(hit.pos - origin);
		return true;
	}
	__device__ bool hit_plane (const Ray& ray, const SurfaceInfo& surface, HitInfo& hit) {
		Vec3 normal = norm(surface.normal);
		Vec3 d = norm(ray.dir);
		float n_dot_dir = dot(normal, d);
		if (n_dot_dir == 0) return false; // ray parallel to plane
		float t = dot(normal, surface.origin - ray.origin) / n_dot_dir;
		if (t < T_EPSILON || t > T_MAX) return false;
		hit.is_hit = true;
		hit.t = t;
		hit.view_dir = d;
		hit.pos = ray.point(t); // dir must be unit vector
		hit.norm = normal;
		return true;	
	}

	__device__ bool hit_triangle(const Ray& ray, const SurfaceInfo& surface, HitInfo& hit) {
		const Vec3& rd = ray.dir;
		const Vec3& ro = ray.origin;
		Vec3& ba = surface.points[1] - surface.points[0];
		Vec3& ca = surface.points[2] - surface.points[0];
		Vec3 normal = norm(cross(ca, ba));
		// Check intersection with the plane
		float n_dot_dir = dot(normal, rd);
		if (n_dot_dir == 0) return false; // ray parallel to plane
		float t = dot(normal, surface.points[0] - ray.origin) / n_dot_dir;
		if (t < T_EPSILON || t > T_MAX) return false;
		// Naive geometric method (anti-clockwise handedness)
		Vec3 p = ray.point(t);
		Vec3 cb = surface.points[2] - surface.points[1];
		Vec3 ac = -ca;
		if (dot(ba, p - surface.points[0]) < 0 ||
			dot(cb, p - surface.points[1]) < 0 ||
			dot(ac, p - surface.points[2]) < 0) return false;
		hit.is_hit = true;
		hit.t = t;
		hit.pos = p;
		hit.view_dir = rd;
		hit.norm = normal;
		return false;
	}

}
