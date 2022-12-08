#include <iostream>
#include "cuda_runtime.h"

#include "Utils.cuh"

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
	/**
	 * Morton bit shifting pre-req for generating morton codes.
	 * Borrowed from PBR book.
	 * 
	 * \param x value to generate into partial morton code from
	 * \return left spaced partial morton code
	 */
	__host__ __device__ uint32_t left_shift_3(uint32_t x) {
		if (x == (1 << 10)) --x;
		x = (x | (x << 16)) & 0b00000011000000000000000011111111;
		x = (x | (x <<  8)) & 0b00000011000000001111000000001111;
		x = (x | (x <<  4)) & 0b00000011000011000011000011000011;
		x = (x | (x <<  2)) & 0b00001001001001001001001001001001;
		return x;
	}
}
