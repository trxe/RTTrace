#ifndef MAT_H
#define MAT_H

#include "Vec3.cuh"

namespace RTTrace {

	struct Mat3 { 
		// Stored in column-major
	public:
		__host__ __device__ Mat3() {};
		__host__ __device__ Mat3(float e1, float e2, float e3, float e4, float e5, float e6, float e7, float e8, float e9) {
			col[0] = { e1, e2, e3 };
			col[1] = { e4, e5, e6 };
			col[2] = { e7, e8, e9 };
		};
		__host__ __device__ Mat3(const Vec3& col1, const Vec3& col2, const Vec3& col3) {
			col[0] = col1;
			col[1] = col2;
			col[2] = col3;
		};

		__host__ __device__ Vec3& operator[](int c) {
			return col[c];
		}

		__host__ __device__ Vec3 operator[](int c) const {
			return col[c];
		}

	private:
		Vec3 col[3];
	};

	__host__ __device__ float det(const Mat3& mat);
}

#endif // !MAT_H
