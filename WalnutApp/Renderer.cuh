#ifndef RENDERER_H
#define RENDERER_H

#include "Light.cuh"
#include "Surface.cuh"
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
		virtual void set_world(SurfaceInfo* surfaces, int count);
		virtual void set_lights(LightInfo* lights, int count);
		virtual void render(float viewport_width, float viewport_height, int recursion_levels, const CameraInfo& info, abgr_t* data) = 0;
	protected:
		SurfaceInfo* surfaces_d;
		int surface_count;
		LightInfo* lights_d;
		int light_count;
		abgr_t* data_d;
	};

	class BasicRaytracer : public Renderer {
	public:
		virtual void render(float viewport_width, float viewport_height, int recursion_levels, const CameraInfo& info, abgr_t* data) override;
	private:
		float last_res[2];
	};
}
#endif 