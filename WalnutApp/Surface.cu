#include "Mat.cuh"
#include "Surface.cuh"
#include "Vec3.cuh"

namespace RTTrace {
	__host__ bool init_bound(SurfaceInfo& surface) {
		AABB& bound = surface.bound;
		switch (surface.type) {
		case SurfaceInfo::PLANE:
			bound.active = false;
			return false;
		case SurfaceInfo::SPHERE:
		{
			Vec3 unit = Vec3(surface.scale, surface.scale, surface.scale);
			bound.minw = surface.origin - unit;
			bound.maxw = surface.origin + unit;
			bound.active = true;
		}
			return true;
		case SurfaceInfo::TRIANGLE:
			bound.minw = surface.points[0];
			bound.maxw = surface.points[0];
			for (int p = 1; p < 3; p++) {
				for (int axis = 0; axis < 3; axis++) {
					bound.minw[axis] = std::min(surface.points[p][axis], bound.minw[axis]);
					bound.maxw[axis] = std::max(surface.points[p][axis], bound.maxw[axis]);
				}
			}
			bound.active = true;
			return true;
		}
	}

	__device__ bool hit_bound(const Ray& r, const SurfaceInfo& surface) {
		const AABB& bound = surface.bound;
		const Vec3& ro = r.origin;
		const Vec3& rd = r.dir;
		if (!bound.active) return true;
		float tming = T_EPSILON, tmaxg = T_MAX;
		for (int axis = 0; axis < 3; axis++) {
			float tmin = (bound.minw[axis] - ro[axis]) / rd[axis];
			float tmax = (bound.maxw[axis] - ro[axis]) / rd[axis];
			if (tmin > tmax) {
				float t = tmin;
				tmin = tmax;
				tmax = t;
			}
			tming = fmaxf(tmin, tming);
			tmaxg = fminf(tmax, tmaxg);
			if (tmaxg < tming) return false;
		}
		return true;
	}

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
		Vec3& ab = surface.points[1] - surface.points[0];
		Vec3& ac = surface.points[2] - surface.points[0];
		Vec3 normal = norm(cross(ac, ab));
		// Naive geometric method (anti-clockwise handedness), faster than Moller-Trombore here
		// Check intersection with the plane
		/*
		float n_dot_dir = dot(normal, rd);
		if (n_dot_dir == 0) return false; // ray parallel to plane
		float t = dot(normal, surface.points[0] - ray.origin) / n_dot_dir;
		if (t < T_EPSILON || t > T_MAX) return false;
		Vec3 p = ray.point(t);
		Vec3 bc = surface.points[2] - surface.points[1];
		Vec3 ca = surface.points[0] - surface.points[2];
		if (dot(cross(ab, p - surface.points[0]), normal) < 0 || 
			dot(cross(bc, p - surface.points[1]), normal) < 0 ||
			dot(cross(-ac, p - surface.points[2]), normal) < 0) return false;
		*/

		// Moller-Trombore (uses Cramer's rule): Benchmark: 7 FPS
		Vec3 aro = ro - surface.points[0];
		Vec3 d = -rd;
		float detA = det(d, ab, ac);
		float t = det(aro, ab, ac) / detA;
		float beta = det(d, aro, ac) / detA;
		float gamma = det(d, ab, aro) / detA;
		if (t < T_EPSILON || t > T_MAX) return false;
		if (beta < 0 || gamma < 0 || beta + gamma > 1) return false;
		Vec3 p = surface.points[0] * (1 - beta - gamma) + surface.points[1] * beta + surface.points[2] * gamma;
		hit.is_hit = true;
		hit.t = t;
		hit.pos = p;
		hit.view_dir = rd;
		hit.norm = normal;
		return true;
	}

}
