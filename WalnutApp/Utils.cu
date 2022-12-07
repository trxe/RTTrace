#include <iostream>
#include "cuda_runtime.h"
#include "Utils.cuh"
#include "Vec3.cuh"

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
	if (result) {
		std::cerr << "CUDA error: " << static_cast<unsigned int>(result) << " => " << cudaGetErrorString(result) << "\nat " <<
		file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

namespace RTTrace {
	// borrowed from pbr-book.
	__host__ __device__ uint32_t left_shift_3(uint32_t x) {
		if (x == (1 << 10)) --x;
		x = (x | (x << 16)) & 0b00000011000000000000000011111111;
		x = (x | (x <<  8)) & 0b00000011000000001111000000001111;
		x = (x | (x <<  4)) & 0b00000011000011000011000011000011;
		x = (x | (x <<  2)) & 0b00001001001001001001001001001001;
		return x;
	}

	__host__ __device__ uint32_t generate_morton_code(Vec3 pos, Vec3 global_min, Vec3 global_max) {
		uint32_t morton = 0x0;
		Vec3 rel_pos = inverse_lerp(pos, global_min, global_max);
		for (int i = 0; i < 3; i++) {
			uint32_t v = fminf(fmaxf(0.0, pos[i] * 1024.0), 1023.0);
			uint32_t vv = left_shift_3(v);
			morton += vv << (2 - i);
		}
		return morton;
	}
}
