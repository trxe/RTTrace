#include "Walnut/Application.h"
#include "Walnut/EntryPoint.h"

#include "Walnut/Image.h"
#include "Walnut/Random.h"
#include "Walnut/Timer.h"

#include "../imgui-filebrowser/imfilebrowser.h"

#include "Demo.h"
#include "FileReader.h"
#include "InputHandler.h"

#include "Light.cuh"
#include "Mat.cuh"
#include "Renderer.cuh"
#include "Surface.cuh"
#include "Utils.cuh"

using namespace Walnut;
using namespace RTTrace;

class RaytracerLayer : public Layer
{
public:
	virtual void OnUIRender() override
	{

		ImGui::Begin("Controls");
		const glm::vec3 pos = input.get_position();
		ImGui::Text("Camera world pos: %.2f %.2f %.2f", pos.x, pos.y, pos.z);
		const glm::vec3 fwd = input.get_forward_dir();
		ImGui::Text("Foward world pos: %.2f %.2f %.2f", fwd.x, fwd.y, fwd.z);
		const glm::vec3 up = input.get_up_dir();
		ImGui::Text("Up world pos: %.2f %.2f %.2f", up.x, up.y, up.z);

		if (ImGui::Button("Select file to load into scene")) {
			file_dialog.SetTypeFilters({ ".fbx" });
			file_dialog_active = true;
			file_dialog.Open();
		}
		if (selected_file.length() > 0) {
			ImGui::TextWrapped("Selected file: %s", selected_file.c_str());
			if (ImGui::Button("Load file")) {
				LoadFile();
			}
		}

		ImGui::InputScalar("Recursion Depth", ImGuiDataType_U32, &recursion_levels, NULL, NULL, "%d");

		if (ImGui::Button("Render")) {
			Render();
		}
		if (last_render_time >= 0) {
			ImGui::Text("Render Time: %.3f ms", last_render_time);
			ImGui::Text("FPS: %d", (int)(1000.0 / last_render_time));
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

		/*
		*/
		if(file_dialog_active) {
			file_dialog.Display();
			
			if(file_dialog.HasSelected()) {
				selected_file = file_dialog.GetSelected().string();

				std::cout << "Selected filename" << selected_file << std::endl;
				file_dialog.ClearSelected();
			}
        }

	}

	virtual void OnUpdate(float ts) override {
		if (input.OnUpdate(ts)) {
			Render();
		}
	}

private:
	ImGui::FileBrowser file_dialog;
	bool file_dialog_active = false;
	std::string selected_file = "";
	InputHandler input;
	std::shared_ptr<Image> m_Image;
	float viewport_width{};
	float viewport_height{};

	float last_render_time = -1;
	abgr_t* data = nullptr;
	AABB global_bound;
	int surface_count;
	int light_count;
	int recursion_levels = 2;
	BasicRaytracer tracer;

	void LoadFile() {
		FBXReader reader;
		std::vector<SurfaceInfo> surface_infos;
		std::vector<LightInfo> light_infos;
		reader.parse(selected_file.c_str(), surface_infos, global_bound);
		tracer.set_world(surface_infos.data(), surface_infos.size());
		light_infos.clear();
		light_infos.resize(1);
		LightInfo& l0 = light_infos[0];
		l0.type = LightInfo::POINT;
		l0.origin = global_bound.maxw + Vec3(1.0, 1.0, 1.0);
		l0.color = Vec3(1.0, 1.0, 1.0);
		l0.intensity = 1.0f;
		light_count = 1;
		tracer.set_lights(light_infos.data(), light_infos.size());
		Render();
	}

	void Render() {
		m_Image = std::make_shared<Image>(viewport_width, viewport_height, ImageFormat::RGBA);
		size_t pixel_count = static_cast<size_t>(viewport_width * viewport_height);

		if (last_render_time < 0) {
			std::vector<SurfaceInfo> surface_infos;
			std::vector<LightInfo> light_infos;
			load_demo_scene(surface_infos, light_infos, global_bound);
			tracer.set_world(surface_infos.data(), surface_infos.size());
			tracer.set_lights(light_infos.data(), light_infos.size());
		}

		Timer timer;

		delete[] data;

		data = new abgr_t[pixel_count];

		tracer.render(viewport_width, viewport_height, recursion_levels, input.get_cam_info(), data);

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
