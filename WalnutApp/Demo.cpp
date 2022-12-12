#include "Demo.h"

#include "BoundingVolume.cuh"
#include "Surface.cuh"
#include "Light.cuh"

namespace RTTrace {
	__host__ void load_demo_scene(std::vector<SurfaceInfo>& surfaces, std::vector<LightInfo>& lights, AABB& global_bound) {
		surfaces.clear();
		surfaces.resize(3);
		SurfaceInfo& s0 = surfaces[0];
		s0.type = SurfaceInfo::SPHERE;
		s0.origin = Vec3(1.0f, 1.0f, -1.0f);
		s0.scale = 1.4f;
		s0.mat.ka = Vec3(0.2f, 0.0f, 0.0f);
		s0.mat.kd = Vec3(1.0f, 0.7f, 0.0f);
		s0.mat.ks = Vec3(1.0f, 1.0f, 1.0f);
		s0.mat.krg = Vec3(0.3f, 0.3f, 0.3f);
		s0.mat.n = 128;
		SurfaceInfo& s1 = surfaces[1];
		s1.type = SurfaceInfo::TRIANGLE;
		s1.points[0] = Vec3(0.0f, -0.1f, 1.3f);
		s1.points[1] = Vec3(0.0f, 0.1f, 0.0f);
		s1.points[2] = Vec3(1.3f, 0.1f, 0.0f);
		s1.normal = Vec3(0.0f, 1.0f, 0.0f);
		s1.mat.ka = Vec3(0.1f, 0.0f, 0.0f);
		s1.mat.kd = Vec3(0.7f, 0.0f, 0.4f);
		s1.mat.ks = Vec3(1.0f, 1.0f, 1.0f);
		s1.mat.krg = Vec3(0.8f, 0.8f, 0.8f);
		s1.mat.n = 64;
		/*
		SurfaceInfo& s1 = surface_infos[1];
		s1.type = SurfaceInfo::SPHERE;
		s1.origin = Vec3(1.0f, 1.0f, 2.0f);
		s1.scale = 0.4;
		s1.mat.ka = Vec3(0.0f, 0.3f, 0.0f);
		s1.mat.kd = Vec3(0.0f, 1.0f, 0.4);
		s1.mat.ks = Vec3(1.0f, 1.0f, 1.0f);
		s1.mat.n = 64;
		*/
		SurfaceInfo& s2 = surfaces[2];
		s2.type = SurfaceInfo::PLANE;
		s2.origin = Vec3(0.0f, -2.0f, -2.0f);
		s2.normal = Vec3(0.0f, 2.0f, 0.0f);
		s2.mat.ka = Vec3(0.0f, 0.0f, 0.1f);
		s2.mat.kd = Vec3(0.0f, 0.2f, 0.5f);
		s2.mat.ks = Vec3(1.0f, 1.0f, 1.0f);
		s2.mat.krg = Vec3(0.8f, 0.8f, 0.8f);
		s2.mat.n = 64;

		lights.clear();
		lights.resize(1);
		LightInfo& l0 = lights[0];
		l0.type = LightInfo::POINT;
		l0.origin = Vec3(1.0f, 3.0f, 2.0f);
		l0.color = Vec3(1.0f, 1.0f, 1.0f);
		l0.intensity = 1.0f;
	}
}
