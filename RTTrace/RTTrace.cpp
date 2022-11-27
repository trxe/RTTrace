#include <iostream>
#include <memory>

#include "Walnut/Application.h"
#include "Walnut/EntryPoint.h"

#include "Walnut/Image.h"
#include "Walnut/Random.h"
#include "Walnut/Timer.h"

#include "Camera.h"
#include "Surface.h"
#include "Util.h"

using namespace Walnut;
using namespace RTTrace;
using abgr_t = uint32_t;

class RaytracerLayer : public Layer
{
public:
	virtual void OnUIRender() override
	{
		// Controls
		ImGui::Begin("Controls");
		if (ImGui::Button("Render")) {
			Render();
		}
		if (last_render_time >= 0) {
			ImGui::Text("Render Time: %.3f ms", last_render_time);
		}
		ImGui::End();

		// Viewport
		ImGui::Begin("Viewport");
		viewport_width = ImGui::GetContentRegionAvail().x;
		viewport_height = ImGui::GetContentRegionAvail().y;
		if (!m_Image || viewport_width != m_Image->GetWidth() || viewport_height != m_Image->GetHeight()) {
			Render();
		}

		if (m_Image) {
			ImGui::Image(m_Image->GetDescriptorSet(), { (float)m_Image->GetWidth(), (float)m_Image->GetHeight() });
		}

		ImGui::End();

	}

private:
	Camera camera{ Vec3{0,0,6} };
	std::vector<shared_ptr<Surface>> surfaces = generate_default_surfaces();
	std::shared_ptr<Image> m_Image;
	abgr_t* data;
	float viewport_width{};
	float viewport_height{};

	float last_render_time = -1;

	void Render()
	{
		float pixel_count = viewport_width * viewport_height;
		delete[] data;
		data = new abgr_t[pixel_count];
		m_Image = std::make_shared<Image>(viewport_width, viewport_height, ImageFormat::RGBA);

		Timer timer;

		camera.perspective(45, Vec3(0,0,0), viewport_width, viewport_height);
		for (float y = 0; y < viewport_height; y += 1.0) {
			for (float x = 0; x < viewport_width; x += 1.0) {
				Ray r = camera.gen_ray(x, y);
				int index = (int)y * viewport_width + (int)x;
				HitInfo hit{};
				for (auto &surface : surfaces) {
					HitInfo this_hit;
					bool is_hit = surface->hit(r, this_hit);
					if (is_hit) {
						hit = (hit.t < 0 || this_hit.t < hit.t) ? this_hit : hit;
					}
				}
				data[index] = hit.t > 0 ? Vec3ToARGB(glm::normalize(r.dir)) : 0x0;
				/* */
			}
		}

		last_render_time = timer.ElapsedMillis();

		m_Image->SetData(data);
	}

};

Walnut::Application* Walnut::CreateApplication(int argc, char** argv)
{
	Walnut::ApplicationSpecification spec;
	spec.Name = "Walnut Example";

	Walnut::Application* app = new Walnut::Application(spec);
	app->PushLayer<RaytracerLayer>();
	app->SetMenubarCallback([app]()
		{
			if (ImGui::BeginMenu("File"))
			{
				if (ImGui::MenuItem("Exit"))
				{
					app->Close();
				}
				ImGui::EndMenu();
			}
		});
	return app;
}