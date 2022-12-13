#ifndef BOUND_H
#define BOUND_H

#include "cuda_runtime.h"

#include "Surface.cuh"
#include "Vec3.cuh"

namespace RTTrace {
	struct AABB {
		bool active = false;
		Vec3 minw;
		Vec3 maxw;
		size_t left_child_bound_idx;
		size_t right_child_bound_idx;
		// SurfaceInfo* surface_device{ nullptr };
		__host__ __device__ bool is_surface() const { return left_child_bound_idx == right_child_bound_idx; }
		int surface_idx{ -1 };
	};

}

#endif // !BOUND_H
