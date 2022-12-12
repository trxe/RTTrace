#include "Mat.cuh"
#include "Surface.cuh"
#include "Vec3.cuh"

namespace RTTrace {
	// __host__ __device__ bool init_bound(SurfaceInfo& surface) {
	__device__ bool init_bound(SurfaceInfo& surface) {
		switch (surface.type) {
		case SurfaceInfo::PLANE:
			surface.origin *= surface.scale;
			surface.scale = 1.0;
			return false;
		case SurfaceInfo::SPHERE:
			surface.origin *= surface.scale;
			Vec3 unit = Vec3(surface.scale, surface.scale, surface.scale);
			surface.minw = surface.origin - unit;
			surface.maxw = surface.origin + unit;
			return true;
		case SurfaceInfo::TRIANGLE:
			for (int p = 0; p < 3; p++) {
				surface.points[p] *= surface.scale;
			}
			surface.minw = surface.points[0];
			surface.maxw = surface.points[0];
			surface.origin = surface.points[0];
			for (int p = 1; p < 3; p++) {
				for (int axis = 0; axis < 3; axis++) {
					surface.minw[axis] = fminf(surface.points[p][axis], surface.minw[axis]);
					surface.maxw[axis] = fmaxf(surface.points[p][axis], surface.maxw[axis]);
				}
				surface.origin += surface.points[p];
			}
			surface.origin *= 1.0f / 3.0f;
			surface.scale = 1.0;
			return true;
		}
	}

	__device__ bool hit_bound(const Ray& r, const Vec3& minw, const Vec3& maxw) {
		const Vec3& ro = r.origin;
		const Vec3& rd = r.dir;
		float tming = T_EPSILON, tmaxg = T_MAX;
		for (int axis = 0; axis < 3; axis++) {
			if (minw[axis] == -INFINITY || maxw[axis] == INFINITY) { 
				return true;
			}
			float tmin = (minw[axis] - ro[axis]) / rd[axis];
			float tmax = (maxw[axis] - ro[axis]) / rd[axis];
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
