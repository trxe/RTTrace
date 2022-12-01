#include "Walnut/Application.h"
#include "Walnut/EntryPoint.h"

#include "Walnut/Image.h"
#include "Walnut/Random.h"
#include "Walnut/Timer.h"

#include "InputHandler.h"
#include "Renderer.cuh"
#include "Utils.cuh"

using namespace Walnut;

class RaytracerLayer : public Layer
{
public:
	virtual void OnUIRender() override
	{
		// Controls
		ImGui::Begin("Controls");
		const glm::vec3 pos = input.get_position();
		ImGui::Text("Camera world pos: %.2f %.2f %.2f", pos.x, pos.y, pos.z);
		const glm::vec3 fwd = input.get_forward_dir();
		ImGui::Text("Foward world pos: %.2f %.2f %.2f", fwd.x, fwd.y, fwd.z);
		const glm::vec3 up = input.get_up_dir();
		ImGui::Text("Up world pos: %.2f %.2f %.2f", up.x, up.y, up.z);

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
		if (viewport_height != 0 && viewport_width != 0) {
			if (!m_Image || viewport_width != m_Image->GetWidth() || viewport_height != m_Image->GetHeight()) {
				Render();
			}

			if (m_Image) {
				ImGui::Image(m_Image->GetDescriptorSet(), { (float)m_Image->GetWidth(), (float)m_Image->GetHeight() });
			}
		}

		ImGui::End();

	}

	virtual void OnUpdate(float ts) override {
		input.OnUpdate(ts);
	}

private:
	InputHandler input;
	std::shared_ptr<Image> m_Image;
	float viewport_width{};
	float viewport_height{};

	float last_render_time = -1;
	abgr_t* data = nullptr;
	RTTrace::BasicRaytracer tracer;

	void Render()
	{
		m_Image = std::make_shared<Image>(viewport_width, viewport_height, ImageFormat::RGBA);
		size_t pixel_count = static_cast<size_t>(viewport_width * viewport_height);

		Timer timer;
		// abgr_t* data = renderer.render(viewport_width, viewport_height);
		delete[] data;

		data = new abgr_t[pixel_count];

		// PLACEHOLDER RED
		for (size_t i{}; i < pixel_count; i++) {
			data[i] = 0xff0000ff;
		}
		tracer.render(viewport_height, viewport_width, input.get_cam_info(), data);

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
