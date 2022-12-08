#include <iostream>
#include <vector>
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

