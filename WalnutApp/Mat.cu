#include "Mat.cuh"

namespace RTTrace {

	__host__ __device__ float det(const Mat3& mat) {
		float m[3][3];
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				m[i][j] = mat[j][i];
			}
		}
		float d = 0;
		d += m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]);
		d -= m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]);
		d += m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
		return d;
	}

}