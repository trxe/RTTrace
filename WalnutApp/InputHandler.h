#ifndef INPUTHANDLER_H
#define INPUTHANDLER_H

#include "Utils.cuh"

/**
 * Input handler based on TheCherno's Interactive camera system from https://youtu.be/lXlXqUEEJ94.
 */
class InputHandler {
public:
	bool OnUpdate(float ts);
	float get_rotation_speed() { return 0.3f; }

	const RTTrace::CameraInfo get_cam_info() { return camInfo; }

	glm::vec3 get_position() {
		return glm::vec3(camInfo.eye[0], camInfo.eye[1], camInfo.eye[2]);
	}

	glm::vec3 get_up_dir() {
		float* up = camInfo.up;
		return glm::vec3(up[0], up[1], up[2]);
	}

	glm::vec3 get_forward_dir() {
		float* at = camInfo.at;
		float* eye = camInfo.eye;
		return glm::vec3(at[0] - eye[0], at[1] - eye[1], at[2] - eye[2]);
	}

private:
	RTTrace::CameraInfo camInfo{};
	glm::vec2 m_LastMousePosition{};
};

#endif
