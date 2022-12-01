#ifndef RENDERER_H
#define RENDERER_H

#include "Utils.cuh"

namespace RTTrace {
	/**
	 * Base class for all renderers. Called by Main.cpp to render to display
	 */
	class Renderer {
	public:
		/**
		 * Renders to the display.
		 * 
		 * \param viewport_width
		 * \param viewport_height
		 * \param info Camera information payload to be sent to kernel to create Camera
		 * \param data Image data in abgr_t format
		 * \return 
		 */
		virtual void render(float viewport_width, float viewport_height, const CameraInfo& info, abgr_t* data) = 0;
	};

	class BasicRaytracer : public Renderer {
	public:
		virtual void render(float viewport_width, float viewport_height, const CameraInfo& info, abgr_t* data) override;
	};
}
#endif 