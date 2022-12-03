#include "glm/gtc/quaternion.hpp"
#include "glm/gtx/quaternion.hpp"
#include "glm/glm.hpp"

#include "Walnut/Input/Input.h"

#include "InputHandler.h"

using namespace Walnut;

bool InputHandler::OnUpdate(float ts) {
	glm::vec2 mousePos = Input::GetMousePosition();
	glm::vec2 delta = (mousePos - m_LastMousePosition) * 0.002f;
	m_LastMousePosition = mousePos;

	// Reset
	if (Input::IsKeyDown(KeyCode::R))
	{
		camInfo = {};
		return true;
	}

	if (!Input::IsMouseButtonDown(MouseButton::Right))
	{
		Input::SetCursorMode(CursorMode::Normal);
		return false;
	}

	Input::SetCursorMode(CursorMode::Locked);

	bool moved = false;

	glm::vec3 m_Position = get_position();
	glm::vec3 m_UpDirection = get_up_dir();
	glm::vec3 m_ForwardDirection = get_forward_dir();
	glm::vec3 rightDirection = glm::cross(m_UpDirection, m_ForwardDirection);

	float speed = 5.0f;

	// Movement
	if (Input::IsKeyDown(KeyCode::W))
	{
		m_Position += m_ForwardDirection * speed * ts;
		moved = true;
	}
	else if (Input::IsKeyDown(KeyCode::S))
	{
		m_Position -= m_ForwardDirection * speed * ts;
		moved = true;
	}
	if (Input::IsKeyDown(KeyCode::A))
	{
		m_Position -= rightDirection * speed * ts;
		moved = true;
	}
	else if (Input::IsKeyDown(KeyCode::D))
	{
		m_Position += rightDirection * speed * ts;
		moved = true;
	}
	if (Input::IsKeyDown(KeyCode::Q))
	{
		m_Position += m_UpDirection * speed * ts;
		moved = true;
	}
	else if (Input::IsKeyDown(KeyCode::E))
	{
		m_Position -= m_UpDirection * speed * ts;
		moved = true;
	}

	// Rotation
	if (delta.x != 0.0f || delta.y != 0.0f)
	{
		float pitchDelta = delta.y * get_rotation_speed();
		float yawDelta = delta.x * get_rotation_speed();

		glm::quat q = glm::normalize(glm::cross(glm::angleAxis(pitchDelta, rightDirection),
			glm::angleAxis(-yawDelta, glm::vec3(0.f, 1.0f, 0.0f))));
		m_ForwardDirection = glm::rotate(q, m_ForwardDirection);
		m_UpDirection = glm::rotate(q, m_UpDirection);

		moved = true;
	}

	if (moved) {
		camInfo.eye[0] = m_Position.x;
		camInfo.eye[1] = m_Position.y;
		camInfo.eye[2] = m_Position.z;

		camInfo.at[0] = m_Position.x + m_ForwardDirection.x;
		camInfo.at[1] = m_Position.y + m_ForwardDirection.y;
		camInfo.at[2] = m_Position.z + m_ForwardDirection.z;

		camInfo.up[0] = m_UpDirection.x;
		camInfo.up[1] = m_UpDirection.y;
		camInfo.up[2] = m_UpDirection.z;

		// RecalculateView();
		return true;
	}

	return false;
}
