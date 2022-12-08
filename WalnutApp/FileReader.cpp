#include <iostream>

#include <fbxsdk.h>

#include "FileReader.h"

using namespace fbxsdk;

namespace RTTrace {

	bool FBXReader::parse(const char* filename, std::vector<SurfaceInfo>& surface_infos, AABB& global_box) {
		FbxImporter* importer = FbxImporter::Create(fbx_manager, "");
		bool import_success = importer->Initialize(filename, -1, fbx_manager->GetIOSettings());
		if (!import_success) {
			std::cerr << "Call to FBX Importer failed: " << importer->GetStatus().GetErrorString() << std::endl;
			importer->Destroy();
			return false;
		}
		FbxScene* scene = FbxScene::Create(fbx_manager, "Scene To Import");
		if (!scene) {
			std::cerr << "Scene creation failed: " << importer->GetStatus().GetErrorString() << std::endl;
			return false;
		}
		importer->Import(scene);
		importer->Destroy();
		FbxNode* root = scene->GetRootNode();
		if (!root) {
			std::cerr << "No root node found in scene " << scene->GetName() << std::endl;
		}

		const int num_nodes = root->GetChildCount();
		for (int n = 0; n < num_nodes; n++) {
			FbxNode* node = root->GetChild(n);
			if (!node) continue;
			FbxMesh* mesh = node->GetMesh();
			if (mesh) {
				mesh->RemoveBadPolygons();
				// Triangulate the mesh
				FbxGeometryConverter geom(fbx_manager);
				mesh = static_cast<FbxMesh*>(geom.Triangulate(mesh, true));
				FbxVector4* fbx_pts = mesh->GetControlPoints();
				int fbx_pts_count = mesh->GetControlPointsCount();
				int fbx_tri_count = mesh->GetPolygonCount();
				int fbx_tri_vert_count = mesh->GetPolygonVertexCount();
				std::cout << "Number of Triangles in " << filename << ": " << fbx_tri_count << std::endl;
				surface_infos.clear();
				surface_infos.resize(fbx_tri_count);
				size_t v{0};
				for (int i = 0; i < fbx_tri_count; i++) {
					if (mesh->GetPolygonSize(i) != 3) {
						std::cerr << "Non-triangle detected in Node " << n << ", Polygon " << i << std::endl;
						break;
					}
					surface_infos[i].type = SurfaceInfo::TRIANGLE;
					for (int j = 0; j < 3; j++, v++) {
						auto pt = fbx_pts[mesh->GetPolygonVertex(i, j)];
						surface_infos[i].points[2-j] = { 
							static_cast<float>(pt[0]), 
							static_cast<float>(pt[1]), 
							static_cast<float>(pt[2])
						};
					}
					surface_infos[i].mat.ka = Vec3(0.0, 0.2, 0.1);
					surface_infos[i].mat.kd = Vec3(0.0, 0.8, 0.4);
					surface_infos[i].mat.ks = Vec3(1.0, 1.0, 1.0);
					surface_infos[i].mat.krg = Vec3(0.8, 0.8, 0.8);
					surface_infos[i].mat.n = 4.0;
				}
				mesh->ComputeBBox();
				auto bmin = mesh->BBoxMin.Get();
				auto bmax = mesh->BBoxMax.Get();
				global_box.minw = { static_cast<float>(bmin[0]), static_cast<float>(bmin[1]), static_cast<float>(bmin[2]) };
				global_box.maxw = { static_cast<float>(bmax[0]), static_cast<float>(bmax[1]), static_cast<float>(bmax[2]) };
			}

		}

		scene->Destroy();
		return true;
	}
}
