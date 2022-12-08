#ifndef FBX_READER_H
#define FBX_READER_H

#include <fbxsdk.h>

#include <fstream>
#include <vector>
#include "Surface.cuh"

#if _DEBUG 
#pragma comment (lib, "C:\\Program Files\\Autodesk\\FBX\\FBX SDK\\2020.2\\lib\\vs2019\\x64\\debug\\libfbxsdk-md.lib")
#pragma comment (lib, "C:\\Program Files\\Autodesk\\FBX\\FBX SDK\\2020.2\\lib\\vs2019\\x64\\debug\\libxml2-md.lib")
#pragma comment (lib, "C:\\Program Files\\Autodesk\\FBX\\FBX SDK\\2020.2\\lib\\vs2019\\x64\\debug\\zlib-md.lib")
#else // for RELEASE
#pragma comment (lib, "C:\\Program Files\\Autodesk\\FBX\\FBX SDK\\2020.2\\lib\\vs2019\\x64\\release\\libfbxsdk-md.lib")
#pragma comment (lib, "C:\\Program Files\\Autodesk\\FBX\\FBX SDK\\2020.2\\lib\\vs2019\\x64\\release\\libxml2-md.lib")
#pragma comment (lib, "C:\\Program Files\\Autodesk\\FBX\\FBX SDK\\2020.2\\lib\\vs2019\\x64\\release\\zlib-md.lib")
#endif

using namespace fbxsdk;

namespace RTTrace {

	class FileReader {
	public:
		/**
		 * Returns true if success at loading, and fills surface_infos with the 
		 * 
		 * \param filename FBX to load
		 * \param surface_infos Vector to fill with loaded triangle infos.
		 * \return Success/failure to load file
		 */
		virtual bool parse(const char* filename, std::vector<SurfaceInfo>& surface_infos, AABB& global_box) = 0;

	};

	class FBXReader : public FileReader {
	public:
		FBXReader() {
			fbx_manager = FbxManager::Create();
			fbx_io_settings = FbxIOSettings::Create(fbx_manager, IOSROOT);
			fbx_manager->SetIOSettings(fbx_io_settings);
		}

		~FBXReader() {
			fbx_io_settings->Destroy();
			fbx_manager->Destroy();
		}

		virtual bool parse(const char* filename, std::vector<SurfaceInfo>& surface_infos, AABB& global_box) override;

		constexpr bool is_valid() const { return fbx_manager != nullptr; }

	private:
		FbxManager* fbx_manager{ nullptr };
		FbxIOSettings* fbx_io_settings{ nullptr };
	};

}

#endif // !FBX_READER_H

