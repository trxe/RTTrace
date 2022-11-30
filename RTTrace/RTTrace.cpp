#include <iostream>
#include <memory>

#include "Walnut/Application.h"
#include "Walnut/EntryPoint.h"

#include "Walnut/Image.h"
#include "Walnut/Random.h"
#include "Walnut/Timer.h"

#include "Camera.cuh"
#include "Surface.h"
#include "Util.h"
#include "Renderer.cuh"

using namespace Walnut;
using namespace RTTrace;
using abgr_t = uint32_t;

std::vector<shared_ptr<Surface>> generate_default_surfaces() {
	std::vector<shared_ptr<Surface>> surface_list;
	surface_list.emplace_back(make_shared<Sphere>(Vec3(0.0,0.0,0.0), 1.2));
	surface_list.emplace_back(make_shared<Plane>(Vec3(0.0,-2.0, 0.0), Vec3(0.0, 1.0, 0.0)));
	return surface_list;
}

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
	float viewport_width{};
	float viewport_height{};
	Renderer renderer{ camera, surfaces };

	float last_render_time = -1;

	void Render()
	{
		m_Image = std::make_shared<Image>(viewport_width, viewport_height, ImageFormat::RGBA);

		Timer timer;
		abgr_t* data = renderer.render(viewport_width, viewport_height);
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