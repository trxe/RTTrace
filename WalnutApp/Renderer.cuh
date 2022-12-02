#ifndef RENDERER_H
#define RENDERER_H

#include "Utils.cuh"
#include "Surface.cuh"

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
		// virtual void set_world(Surface** surfaces, int count);
		virtual void render(float viewport_width, float viewport_height, const CameraInfo& info, abgr_t* data) = 0;
	protected:
		// Surface* surfaces_d;
		// int surface_count;
		abgr_t* data_d;
	};

	class BasicRaytracer : public Renderer {
	public:
		virtual void render(float viewport_width, float viewport_height, const CameraInfo& info, abgr_t* data) override;
	private:
		float last_res[2];
	};
}
#endif 