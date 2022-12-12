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
		SurfaceInfo* surface_device{ nullptr };
		// int surface_idx{ -1 };
	};

	__device__ inline static void merge_bounds(AABB& output, const AABB& left, const AABB& right) {
		output.minw = vmin(left.minw, right.minw);
		output.minw = vmin(output.minw, left.maxw);
		output.minw = vmin(output.minw, right.maxw);
		output.maxw = vmax(left.maxw, right.maxw);
		output.maxw = vmax(output.maxw, left.minw);
		output.maxw = vmax(output.maxw, right.minw);
	}
}

#endif // !BOUND_H
