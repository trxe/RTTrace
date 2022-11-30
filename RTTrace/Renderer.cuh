#include <memory>
#include <vector>

#include "Camera.cuh"
#include "Surface.h"

#ifndef RENDERER_H
#define RENDERER_H

using abgr_t = uint32_t;

namespace RTTrace {
	class Renderer {
	public:
		Renderer() = delete;
		Renderer(Camera& camera, std::vector<shared_ptr<Surface>> surfaces)
			: camera(camera), surfaces(surfaces){};

		abgr_t* const render(float viewport_width, float viewport_height);

	private:
		Camera camera;
		std::vector<shared_ptr<Surface>> surfaces;
		abgr_t* data;
	};

}
#endif 
