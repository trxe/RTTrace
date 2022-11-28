/* From Cherno's Walnut. Original code from:
* https://github.com/TheCherno/Walnut/blob/20f940b9d23946d4836b8549ff3e2c0750c5d985/Walnut/src/Walnut/Input/Input.h
*/
#pragma once

#include "KeyCodes.h"

#include <glm/glm.hpp>

namespace Walnut {

	class Input
	{
	public:
		static bool IsKeyDown(KeyCode keycode);
		static bool IsMouseButtonDown(MouseButton button);

		static glm::vec2 GetMousePosition();

		static void SetCursorMode(CursorMode mode);
	};

}
