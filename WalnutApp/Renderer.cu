#include <bitset>
#include <iostream>

#include "cuda_runtime.h"

#include "BoundingVolume.cuh"
#include "Camera.cuh"
#include "Ray.cuh"
#include "Renderer.cuh"
#include "Surface.cuh"
#include "Utils.cuh"
#include "Vec3.cuh"

#define BLOCK_LENGTH 16
#define ACC_BLOCK_SIZE 512

namespace RTTrace {

	__device__ Vec3 get_light(HitInfo& hit, const LightInfo* lights, int light_count, const SurfaceInfo* surfaces, int surface_count) {
		if (hit.surface_index < 0) return 0x0;
		const SurfaceInfo& surface = surfaces[hit.surface_index];
		// Back face culling
		if (dot(hit.norm, hit.view_dir) > 0) return surface.mat.ka;
		Vec3 color;
		for (int i = 0; i < light_count; i++) {
			const LightInfo& l = lights[i];
			Ray shadow_ray{hit.pos, l.origin - hit.pos};
			HitInfo shadow_hit;
			for (int s = 0; s < surface_count && !shadow_hit.is_hit; s++) {
				// ignore this surface.
				if (s == hit.surface_index) continue;
				if (!hit_bound(shadow_ray, surfaces[i].minw, surfaces[i].maxw)) continue;

				switch (surfaces[s].type) {
				case SurfaceInfo::PLANE:
					hit_plane(shadow_ray, surfaces[s], shadow_hit);
					break;
				case SurfaceInfo::SPHERE:
					hit_sphere(shadow_ray, surfaces[s], shadow_hit);
					break;
				case SurfaceInfo::TRIANGLE:
					hit_triangle(shadow_ray, surfaces[s], shadow_hit);
					break;
				}
			}
			if (!shadow_hit.is_hit) {
				float n_dot_l = dot(hit.norm, shadow_ray.dir);
				if (n_dot_l <= 0.0) continue;
				color += l.color * surface.mat.kd * l.intensity * n_dot_l;
				Vec3 r = reflect(shadow_ray.dir, hit.norm);
				hit.refl_dir = r;
				float r_dot_v = fmaxf(0, dot(r, norm(-hit.view_dir)));
				float r_dot_v_pow_n = powf(r_dot_v, surface.mat.n);
				Vec3 specular = l.color * surface.mat.ks * l.intensity * r_dot_v_pow_n;
				color += specular;
			}
		}
		color += surface.mat.ka;
		return color;
	}

	// On a single thread only
	__device__ int iterative_traversal(int* list, const AABB* bvh, const Ray& ray) {
		size_t aabb_stack[64];
		int surface_index = 0;
		size_t* head = aabb_stack;
		*head++ = 0; // push [0] onto stack
		
		do {
			// ASSUMES HEAD IS ALWAYS HIT
			const AABB& aabb = bvh[*(--head)];
			if (aabb.is_surface()) {
				list[surface_index] = aabb.surface_idx;
				surface_index++;
				continue;
			}
			const AABB& left = bvh[aabb.left_child_bound_idx];
			if (hit_bound(ray, left.minw, left.maxw)) {
				*(head++) = aabb.left_child_bound_idx;
			}
			const AABB& right = bvh[aabb.right_child_bound_idx];
			if (hit_bound(ray, right.minw, right.maxw)) {
				*(head++) = aabb.right_child_bound_idx;
			}
		} while (head != aabb_stack);
		return surface_index;
	}

	__global__ void gpu_acc_render(
			const CameraInfo* bound, int width, int height, abgr_t* data, 
			const AABB* bvh, const SurfaceInfo* surfaces, int surface_count, 
			const LightInfo* lights, int light_count, int recursion_levels) {
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		// need to invert position
		int pos = (height - y) * width + (width - x);
		if (x >= width || y >= height) return;
		Camera c(*bound, width, height);
		Ray ray = c.gen_ray(x, y);
		Vec3 color{};
		Vec3 att = Vec3(1, 1, 1);
		int surface_list[ACC_BLOCK_SIZE];
		for (int r = 0; r <= recursion_levels; r++) {
			HitInfo hit_global;
			const AABB& aabb = bvh[0];
			int aabb_idx = 0;
			if (!hit_bound(ray, aabb.minw, aabb.maxw)) { 
				// printf("%f %f %f | %f %f %f \n", aabb.minw[0], aabb.minw[1], aabb.minw[2], aabb.maxw[0], aabb.maxw[1], aabb.maxw[2]);
				break;
			}
			// printf("SUCC %f %f %f | %f %f %f \n", aabb.minw[0], aabb.minw[1], aabb.minw[2], aabb.maxw[0], aabb.maxw[1], aabb.maxw[2]);
			int surface_count = iterative_traversal(surface_list, bvh, ray);
			for (int i = 0; i < surface_count; i++) {
				// if (x == 0 && y == 0 && i == 0) printf("%f %f %f\n", surfaces[i].origin[0], surfaces[i].origin[1], surfaces[i].origin[2]);
				// if (!hit_bound(ray, surfaces[i].minw, surfaces[i].maxw)) continue;
				HitInfo hit;
				switch (surfaces[i].type) {
				case SurfaceInfo::SPHERE:
					hit_sphere(ray, surfaces[i], hit);
					break;
				case SurfaceInfo::TRIANGLE:
					hit_triangle(ray, surfaces[i], hit);
					break;
				}
				if (hit.t < hit_global.t) {
					hit_global = hit;
					hit_global.surface_index = i;
				}
			}

			if (hit_global.is_hit) {
				color += get_light(hit_global, lights, light_count, surfaces, surface_count) * att;
				// set up next level of recursion: light and reflection ray
				att *= surfaces[hit_global.surface_index].mat.krg;
				ray.origin = hit_global.pos;
				ray.dir = norm(hit_global.refl_dir);
			}
		}
		data[pos] = vec3_to_abgr(color);
	}

	__global__ void gpu_render(
			const CameraInfo* bound, int width, int height, abgr_t* data, 
			const SurfaceInfo* surfaces, int surface_count, 
			const LightInfo* lights, int light_count, int recursion_levels) {
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		// need to invert position
		int pos = (height - y) * width + (width - x);
		if (x >= width || y >= height) return;
		Camera c(*bound, width, height);
		Ray ray = c.gen_ray(x, y);
		Vec3 color{};
		Vec3 att = Vec3(1, 1, 1);
		for (int r = 0; r <= recursion_levels; r++) {
			HitInfo hit_global;
			for (int i = 0; i < surface_count; i++) {
				// if (x == 0 && y == 0 && i == 0) printf("%f %f %f\n", surfaces[i].origin[0], surfaces[i].origin[1], surfaces[i].origin[2]);
				if (!hit_bound(ray, surfaces[i].minw, surfaces[i].maxw)) continue;
				HitInfo hit;
				switch (surfaces[i].type) {
				case SurfaceInfo::PLANE:
					hit_plane(ray, surfaces[i], hit);
					break;
				case SurfaceInfo::SPHERE:
					hit_sphere(ray, surfaces[i], hit);
					break;
				case SurfaceInfo::TRIANGLE:
					hit_triangle(ray, surfaces[i], hit);
					break;
				}
				if (hit.t < hit_global.t) {
					hit_global = hit;
					hit_global.surface_index = i;
				}
			}

			if (hit_global.is_hit) {
				color += get_light(hit_global, lights, light_count, surfaces, surface_count) * att;
				// set up next level of recursion: light and reflection ray
				att *= surfaces[hit_global.surface_index].mat.krg;
				ray.origin = hit_global.pos;
				ray.dir = norm(hit_global.refl_dir);
			}
		}
		data[pos] = vec3_to_abgr(color);
	}

	/**
	 * Kernel for generating and sorting in chunks of BLOCKSIZE.
	 * 
	 * \param surfaces
	 * \param global_aabb
	 * \param morton_g
	 * \param idx_g
	 * \param count
	 * \return 
	 */
	__global__ void gen_morton_kernel(SurfaceInfo* surfaces, AABB* global_aabb, uint32_t* morton_g, uint32_t* idx_g, int count) {
		int surface_id = blockIdx.x * blockDim.x + threadIdx.x;
		int tid = threadIdx.x;
		__shared__ uint32_t morton_codes[ACC_BLOCK_SIZE];
		__shared__ uint32_t sorted_obj_ids[ACC_BLOCK_SIZE];
		SurfaceInfo& surface = surfaces[surface_id];
		// set centroids
		init_bound(surface);
		switch (surface.type) {
		case SurfaceInfo::TRIANGLE:
			Vec3 sum = surface.points[0] + surface.points[1] + surface.points[2];
			surface.origin = sum / 3.0;
			break;
		}

		morton_codes[tid] = surface_id < count && surface.type != SurfaceInfo::PLANE ? 
			generate_morton_code(surface.origin, global_aabb->minw, global_aabb->maxw) : 0xFFFFFFFF;
		sorted_obj_ids[tid] = surface_id < count ? surface_id : 0xFFFFFFFF;
		__syncthreads();
		// SORT odd even mergesort
		for (int p = 1; p <= blockDim.x; p <<= 1) {
			int twop = p << 1;
			for (int k = p; k > 0; k >>= 1) {
				size_t next = tid + k;
				if (next < blockDim.x && next / twop == tid / twop && morton_codes[tid] > morton_codes[next]) {
					swap<uint32_t, uint32_t>(morton_codes, sorted_obj_ids, tid, next);
				}
				__syncthreads();
			}
		}
		if (surface_id < count) {
			morton_g[surface_id] = morton_codes[tid];
			idx_g[surface_id] = sorted_obj_ids[tid];
		}
	}

	/**
	 * Kernel for sorting mortons globally.
	 * 
	 * \param morton_g
	 * \param idx_g
	 * \param stride
	 * \param p
	 * \return 
	 */
	__global__ void sort_morton_kernel(uint32_t* morton_g, uint32_t* idx_g, int stride, int p) {
		int gid = blockIdx.x * blockDim.x + threadIdx.x;
		int pos = (gid << 1) - (gid & (stride - 1));
		int halfp = p >> 1;
		if (stride >= halfp) return;
		int offset = gid & (halfp - 1);
		int smaller = (offset >= stride) ? pos - stride : pos;
		int bigger = (offset >= stride) ? pos : pos + stride;
		if (morton_g[smaller] > morton_g[bigger]) {
			swap<uint32_t, uint32_t>(morton_g, idx_g, smaller, bigger);
		}
	}

	/**
	 * Helper function to generate and sort all morton codes.
	 * 
	 * \param surfaces Surfaces
	 * \param global_aabb The largest bounding volume containing all surfaces
	 * \param morton_d Sorted Morton codes on DEVICE
	 * \param idx_d Sorted surface indexes on DEVICE
	 * \param size total size (2^ceil(log(surface_count))
	 * \param surface_count Number of surfaces
	 */
	void generate_sorted_mortons(SurfaceInfo* surfaces, const AABB& global_aabb, uint32_t* morton_d,
		uint32_t* idx_d, int size, int surface_count) {
#if _DEBUG
		cudaEvent_t start, stop;
		checkCudaErrors(cudaEventCreate(&start));
		checkCudaErrors(cudaEventCreate(&stop));
		checkCudaErrors(cudaEventRecord(start));
#endif
		dim3 grid = std::ceil((float)size / ACC_BLOCK_SIZE);
		dim3 block = ACC_BLOCK_SIZE;
		AABB* aabb_d;
		checkCudaErrors(cudaMalloc(&aabb_d, sizeof(AABB)));
		checkCudaErrors(cudaMemcpy(aabb_d, &global_aabb, sizeof(AABB), cudaMemcpyHostToDevice));
		gen_morton_kernel<<<grid,block>>>(surfaces, aabb_d, morton_d, idx_d, surface_count);
		checkCudaErrors(cudaDeviceSynchronize());
		SurfaceInfo* sh = new SurfaceInfo[surface_count];
		grid = std::ceil(size / (ACC_BLOCK_SIZE << 1));
		for (int p = (ACC_BLOCK_SIZE << 1); p <= size; p <<= 1) {
			for (int stride = p; stride > 0; stride >>= 1) {
				sort_morton_kernel<<<grid, block>>>(morton_d, idx_d, stride, p);
			}
		}
		checkCudaErrors(cudaFree(aabb_d));
#if _DEBUG
		checkCudaErrors(cudaEventRecord(stop));
		checkCudaErrors(cudaEventSynchronize(stop));
		float ms = -1;
		checkCudaErrors(cudaEventElapsedTime(&ms, start, stop));
		std::cout << "Blocks: " << std::ceil((float)size / ACC_BLOCK_SIZE) << "; Threads per block: " << ACC_BLOCK_SIZE << std::endl;
		std::cout << ms << " milliseconds" << std::endl;
#endif
#if DEBUG
		uint32_t* morton_h = new uint32_t[surface_count];
		uint32_t* idx_h = new uint32_t[surface_count];
		checkCudaErrors(cudaMemcpy(sh, surfaces, sizeof(SurfaceInfo) * surface_count, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(morton_h, morton_d, sizeof(uint32_t) * surface_count, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(idx_h, idx_d, sizeof(uint32_t) * surface_count, cudaMemcpyDeviceToHost));
		for (int i = 0; i < surface_count; i++) {
			std::cout << idx_h[i] << "\t" << std::bitset<32>(morton_h[i]) << " " << sh[i].origin << std::endl;
		}
		delete[] morton_h;
		delete[] idx_h;
		delete[] sh;
#endif
	}
	
	__device__ size_t get_split(uint32_t* sorted_mortons, size_t s, size_t e) {
		if (e == s) return s;
		uint32_t starting_diff = sorted_mortons[s] ^ sorted_mortons[e];
		size_t common_leading_bits = __clz(starting_diff);
		size_t split = s;
		size_t stride = e - s;
		do {
			//  halving the stride
			stride = (stride + 1) >> 1;
			size_t new_split = split + stride;
			if (new_split < e) {
				uint32_t split_common_lbits = __clz(sorted_mortons[s] ^ sorted_mortons[new_split]);
				if (split_common_lbits > common_leading_bits) {
					split = new_split;
					common_leading_bits = split_common_lbits;
				}
			}
		} while (stride > 1);
		return split;
	}

	__device__ void merge_bounds(AABB& output, const AABB& left, const AABB& right) {
		output.minw = vmin(left.minw, right.minw);
		output.minw = vmin(output.minw, left.maxw);
		output.minw = vmin(output.minw, right.maxw);
		output.maxw = vmax(left.maxw, right.maxw);
		output.maxw = vmax(output.maxw, left.minw);
		output.maxw = vmax(output.maxw, right.minw);
	}

	/**
	 * Kernel for generating BVH.
	 * 
	 * \param surfaces List of all surfaces (excluding bounds).
	 * \param bvh List of all bounds and surfaces, size = total_nodes.
	 * \param surface_count Count of all surfaces.
	 * \param morton_g Morton codes of each surface, size = surface_count;
	 * \param idx_g Index of corresponding surface of morton code, size = surface_count;
	 * \param range_l Index of left-most surface for this bvh node, size = total_nodes and values are surface ids.
	 * \param range_r Index of right-most surface for this bvh node, size = total_nodes and values are surface ids.
	 * \param tree_head Global counter of first unset bvh node.
	 * \return 
	 */
	__global__ void bvh_kernel(SurfaceInfo* surfaces, AABB* bvh, int surface_count, 
		uint32_t* morton_g, uint32_t* idx_g, size_t* range_l, size_t* range_r, size_t* tree_head) {
		constexpr size_t BOUND_SET = 0xFFFFFFFF;
		int sorted_id = blockIdx.x * blockDim.x + threadIdx.x;
		int total_nodes = (surface_count << 1) - 1;
		bool node_set = false;
		if (sorted_id > total_nodes) return;
		if (sorted_id == 0) { 
			*tree_head = 1;
			range_l[sorted_id] = 0;
			range_r[sorted_id] = surface_count - 1;
		}
		__syncthreads();
		int left_id = -1;
		int right_id = -1;
		while (*tree_head < total_nodes) {
			// if (sorted_id == 0) printf("tree head %d, max nodes count %d\n", (int)*tree_head, total_nodes);
			if (!node_set && sorted_id < *tree_head) {
				// printf("%d: L[%d] R[%d] \n", (int)sorted_id, (int)range_l[sorted_id], (int)range_r[sorted_id]);
				AABB& bound = bvh[sorted_id];
				if (range_l[sorted_id] < range_r[sorted_id]) {
					int start = atomicAdd(tree_head, 2);
					left_id = start;
					right_id = start + 1;
					size_t split = get_split(morton_g, range_l[sorted_id], range_r[sorted_id]);
					// used for setting up the bounds later
					range_l[left_id] = range_l[sorted_id];
					range_r[left_id] = split;
					range_l[right_id] = split + 1;
					range_r[right_id] = range_r[sorted_id];
					// setting bvh
					bound.active = true;
					bound.left_child_bound_idx = left_id;
					bound.right_child_bound_idx = right_id;
					// printf("%d:\n\t-split: %d\n\t-left %d: L[%d] R[%d]\n\t-right %d: L[%d] R[%d]\n", (int)sorted_id, (int)split, (int)left_id, (int)range_l[left_id], (int)range_r[left_id], (int)right_id, (int)range_l[right_id], (int)range_r[right_id]);
					// printf("%d: L[%d] R[%d]\t\tTH%d/%d\n", (int)sorted_id, (int)range_l[sorted_id], (int)range_r[sorted_id], (int)*tree_head, total_nodes);
				} else {
					bound.active = morton_g[sorted_id] != 0xFFFFFFFF;
					size_t surface_id = range_l[sorted_id];
					SurfaceInfo& s = surfaces[idx_g[surface_id]];
					bound.minw = s.minw;
					bound.maxw = s.maxw;
					// bound.surface_device = surfaces+surface_id;
					bound.surface_idx = surface_id;
					bound.left_child_bound_idx = sorted_id;
					bound.right_child_bound_idx = sorted_id;
					// printf("%d: [%d]\t\tTH%d/%d\n", (int)sorted_id, (int)range_l[sorted_id], (int)*tree_head, total_nodes);
				}
				node_set = true;
				printf("%d %d %d %d\n", (int)sorted_id, (int)bound.left_child_bound_idx, (int)bound.right_child_bound_idx, (int)*tree_head);
			}
			__syncthreads();
		}

		if (!node_set && sorted_id < *tree_head) {
			if (range_l[sorted_id] == range_r[sorted_id]) {
				AABB& bound = bvh[sorted_id];
				bound.active = true;
				size_t surface_id = range_l[sorted_id];
				SurfaceInfo& s = surfaces[idx_g[surface_id]];
				bound.minw = s.minw;
				bound.maxw = s.maxw;
				// bound.surface_device = surfaces+surface_id;
				bound.surface_idx = surface_id;
				bound.left_child_bound_idx = sorted_id;
				bound.right_child_bound_idx = sorted_id;
				node_set = true;
				// printf("%d: [%d]\tTH%d/%d\n", (int)sorted_id, (int)range_l[sorted_id], (int)*tree_head, total_nodes);
				printf("%d %d %d %d\n", (int)sorted_id, (int)bound.left_child_bound_idx, (int)bound.right_child_bound_idx, (int)*tree_head);
			}
		}

		// printf("%d: L[%d] R[%d]\n", (int)sorted_id, (int)range_l[sorted_id], (int)range_r[sorted_id]);


		// now we move onto setting up the minw and maxw for each of the internal nodes.
		while (range_l[sorted_id] != BOUND_SET) {
			// printf("CHECK %d\n", sorted_id);
			if (range_l[sorted_id] == range_r[sorted_id]) {
				range_l[sorted_id] = range_r[sorted_id] = BOUND_SET;
			} else if (range_l[left_id] == BOUND_SET && range_r[right_id] == BOUND_SET) {
				merge_bounds(bvh[sorted_id], bvh[left_id], bvh[right_id]);
				range_l[sorted_id] = BOUND_SET;
				range_r[sorted_id] = BOUND_SET;
			}
			__syncthreads();
		}

		__syncthreads();

#if DEBUG
		if (sorted_id != 0) return;
		for (int i = 0; i < total_nodes; i++) {
			AABB& b = bvh[i];
			printf("%d:\t{%f, %f, %f}\t{%f, %f, %f}\n", i, b.minw[0], b.minw[1], b.minw[2], b.maxw[0], b.maxw[1], b.maxw[1]);
			// printf("   \trange: L [%d] \tR [%d]\n", (int)range_l[i], (int)range_r[i]);
			printf("   \tchild: L [%d] \tR [%d]\n", (int)b.left_child_bound_idx, (int)b.right_child_bound_idx);
		}
#endif
	}

	__global__ void help(const AABB* bvh, int count) {
		for (int i = 0; i < count; i++) {
			const AABB& aabb = bvh[i];
			printf("%f %f %f | %f %f %f \n", aabb.minw[0], aabb.minw[1], aabb.minw[2], aabb.maxw[0], aabb.maxw[1], aabb.maxw[2]);
		}
	}

	void generate_bvh(AABB* bvh_d, SurfaceInfo* surfaces, int surface_count, uint32_t* morton_d, uint32_t* idx_d) {
		size_t* range_l;
		size_t* range_r;
		size_t* tree_head;
		int total_nodes = (surface_count << 1) - 1;
		checkCudaErrors(cudaMalloc(&range_l, sizeof(size_t) * total_nodes));
		checkCudaErrors(cudaMalloc(&range_r, sizeof(size_t) * total_nodes));
		checkCudaErrors(cudaMalloc(&tree_head, sizeof(size_t)));
		dim3 grid = ceil((float)(surface_count << 1) / ACC_BLOCK_SIZE);
		bvh_kernel<<<grid, ACC_BLOCK_SIZE>>>(surfaces, bvh_d, surface_count, morton_d, idx_d, range_l, range_r, tree_head);
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaDeviceSynchronize());
		// To be passed in from outside later
		checkCudaErrors(cudaFree(range_l));
		checkCudaErrors(cudaFree(range_r));
		checkCudaErrors(cudaFree(tree_head));
	}

	void Renderer::set_world(SurfaceInfo* surfaces, int count, const AABB& global_bound) {
		surface_count = count;
		if (surfaces_d != nullptr) {
			checkCudaErrors(cudaFree(surfaces_d));
		}
		if (bvh_tree != nullptr) {
			checkCudaErrors(cudaFree(bvh_tree));
		}
		checkCudaErrors(cudaMalloc(&surfaces_d, sizeof(SurfaceInfo) * count));
		checkCudaErrors(cudaMemcpy(surfaces_d, surfaces, sizeof(SurfaceInfo) * surface_count, cudaMemcpyHostToDevice));

		size_t max_nodes = 1 << static_cast<int>(std::ceil(std::log2(count)));

		checkCudaErrors(cudaMalloc(&bvh_tree, sizeof(AABB) * max_nodes));

		uint32_t* morton_d;
		checkCudaErrors(cudaMalloc(&morton_d, sizeof(uint32_t) * max_nodes));
		uint32_t* idx_d;
		checkCudaErrors(cudaMalloc(&idx_d, sizeof(uint32_t) * max_nodes));

		// 1. Generate and Sort the Morton Codes
		generate_sorted_mortons(surfaces_d, global_bound, morton_d, idx_d, max_nodes, count);
		// 2. Generate the bvh in chunks of the block size
		generate_bvh(bvh_tree, surfaces_d, count, morton_d, idx_d);
		// int total_nodes = (surface_count << 1) - 1;
		// help << <1, 1 >> > (bvh_tree, total_nodes);

		checkCudaErrors(cudaFree(morton_d));
		checkCudaErrors(cudaFree(idx_d));
	}

	void Renderer::set_lights(LightInfo* lights, int count) {
		light_count = count;
		if (lights_d != nullptr) {
			checkCudaErrors(cudaFree(lights_d));
		}
		checkCudaErrors(cudaMalloc(&lights_d, sizeof(LightInfo) * light_count));
		checkCudaErrors(cudaMemcpy(lights_d, lights, sizeof(LightInfo) * light_count, cudaMemcpyHostToDevice));
	}

	void BasicRaytracer::render(float viewport_width, float viewport_height, int recursion_levels, const CameraInfo& bound, abgr_t* data) {
		size_t pixel_count = static_cast<size_t>(viewport_width * viewport_height);
		// this condition literally saves 1ms HAHAH
		if (last_res[0] != viewport_width || last_res[1] != viewport_height) {
			last_res[0] = viewport_width;
			last_res[1] = viewport_height;
			checkCudaErrors(cudaFree(data_d));
			checkCudaErrors(cudaMalloc(&data_d, pixel_count * sizeof(abgr_t)));
		}

		// int block_length = cuda
		dim3 gridDim(ceil(viewport_width / BLOCK_LENGTH), ceil(viewport_height / BLOCK_LENGTH));
		dim3 blockDim(BLOCK_LENGTH, BLOCK_LENGTH);

		CameraInfo* info_device;
		checkCudaErrors(cudaMalloc(&info_device, sizeof(CameraInfo)));
		checkCudaErrors(cudaMemcpy(info_device, &bound, sizeof(CameraInfo), cudaMemcpyHostToDevice));

		checkCudaErrors(cudaDeviceSynchronize());
		// gpu_render<<<gridDim,blockDim>>>(info_device, static_cast<int>(viewport_width), static_cast<int>(viewport_height), data_d, surfaces_d, surface_count, lights_d, light_count, recursion_levels);
		// checkCudaErrors(cudaDeviceSynchronize());

		// TO REMOVE try rendering just once
		gpu_acc_render<<<gridDim,blockDim>>>(info_device, static_cast<int>(viewport_width), static_cast<int>(viewport_height), data_d, bvh_tree, surfaces_d, surface_count, lights_d, light_count, recursion_levels);
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaMemcpy(data, data_d, pixel_count * sizeof(abgr_t), cudaMemcpyDeviceToHost));
	}
}
