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

		~Renderer() {
			if (surfaces_d != nullptr) checkCudaErrors(cudaFree(surfaces_d));
			if (lights_d != nullptr) checkCudaErrors(cudaFree(lights_d));
			if (bvh_tree != nullptr) checkCudaErrors(cudaFree(bvh_tree));
			if (data_d != nullptr) checkCudaErrors(cudaFree(data_d));
		}

		/**
		 * Renders to the display.
		 * 
		 * \param viewport_width
		 * \param viewport_height
		 * \param info Camera information payload to be sent to kernel to create Camera
		 * \param data Image data in abgr_t format
		 * \return 
		 */
		virtual void render(float viewport_width, float viewport_height, int recursion_levels, const CameraInfo& bound, abgr_t* data) = 0;

		/**
		 * Blocking call to send all world shapes/surfaces to global memory. Non-blocking feature WIP.
		 * 
		 * \param surfaces Array of surfaces
		 * \param count Number of surfaces to send
		 */
		virtual void set_world(SurfaceInfo* surfaces, int count, const AABB& global_bound);

		/**
		 * Blocking call to send all lights to global memory. Only supports point lights currently.
		 * Non-blocking call and area lights WIP.
		 * 
		 * \param lights Array of lights
		 * \param count Number of lights to send
		 */
		virtual void set_lights(LightInfo* lights, int count);

	protected:
		SurfaceInfo* surfaces_d;
		AABB* bvh_tree;
		int surface_count;
		LightInfo* lights_d;
		int light_count;
		abgr_t* data_d;
	};

	class BasicRaytracer : public Renderer {
	public:
		virtual void render(float viewport_width, float viewport_height, int recursion_levels, const CameraInfo& bound, abgr_t* data) override;
	private:
		float last_res[2];
	};
}
#endif 