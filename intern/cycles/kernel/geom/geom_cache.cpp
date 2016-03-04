/*
 * Copyright 2011-2016 Blender Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "util_lru.h"

#include "scene.h"
#include "object.h"
#include "mesh.h"

#include "osl_shader.h"

#include "kernel_compat_cpu.h"
#include "kernel_globals.h"
#include "kernel_random.h"
#include "kernel_projection.h"
#include "kernel_differential.h"
#include "kernel_montecarlo.h"
#include "kernel_camera.h"

#include "geom/geom.h"

#include "kernel_accumulate.h"
#include "kernel_shader.h"

CCL_NAMESPACE_BEGIN

struct GeomCacheThreadData;
struct SubPatchWraper;

typedef struct GeomCache {
	struct patch_key_t {
		uint object, prim;

		patch_key_t() : object(0), prim(0) {}
		patch_key_t(uint object, uint key) : object(object), prim(key) {}

		bool operator == (const patch_key_t& other) const {
			return (object == other.object) && (prim == other.prim);
		}

		struct hasher_t {
			size_t operator()(const patch_key_t& key) const {
				return hash_int_2d(key.object, key.prim);
			}
		};
	};

	typedef LRU<patch_key_t, SubPatchWraper, patch_key_t::hasher_t> lru_t;
	lru_t lru;

	Scene* scene;
	thread_specific_ptr<GeomCacheThreadData> thread_data;

	GeomCacheThreadData* get_thread_data();
} GeomCache;

typedef struct GeomCacheThreadData {
	GeomCache::lru_t::thread_data_t* lru_tdata;
} GeomCacheThreadData;

GeomCacheThreadData* GeomCache::get_thread_data() {
	GeomCacheThreadData* tdata = thread_data.get();
	if(!tdata) {
		tdata = new GeomCacheThreadData;
		thread_data.reset(tdata);

		tdata->lru_tdata = lru.get_thread_data();
	}
	return tdata;
}

struct SubPatchWraper {
	GeomCache::lru_t::ref_t::intrusive_ref_t intrusive_ref;
	TessellatedSubPatch subpatch;
};

GeomCache* geom_cache_create()
{
	return new GeomCache();
}

void geom_cache_release(GeomCache* geom_cache)
{
	delete geom_cache;
}

void geom_cache_thread_init(KernelGlobals *kg, GeomCache* geom_cache)
{
	kg->geom_cache = geom_cache;
	kg->geom_cache_tdata = geom_cache->get_thread_data();
}

void geom_cache_set_scene(GeomCache* geom_cache, void* scene)
{
	if(geom_cache)
		geom_cache->scene = (Scene*)scene;
}

void geom_cache_clear(GeomCache* geom_cache) {
	if(!geom_cache)
		return;

	GeomCache::lru_t* lru = &geom_cache->lru;
	lru->clear();
}

void geom_cache_set_max_size(GeomCache * geom_cache, uint max_size) {
	if(!geom_cache)
		return;

	GeomCache::lru_t* lru = &geom_cache->lru;
	lru->set_max_size(max_size);
}

static void geom_cache_get_tessellated_subpatch_size(GeomCache* geom_cache, int object, int prim, uint* num_verts, uint* num_tris, int* total_size) {
	Mesh* mesh = geom_cache->scene->objects[object]->mesh;
	mesh->diced_subpatch_size(prim, num_verts, num_tris, total_size);
}

static void geom_cache_dice_subpatch(GeomCache* geom_cache, TessellatedSubPatch* subpatch, int object, int prim) {
	Mesh* mesh = geom_cache->scene->objects[object]->mesh;
	mesh->dice_subpatch(subpatch, prim);
}

static void geom_cache_update_subpatch_size(GeomCache* geom_cache, int object, int prim, int size) {
	Mesh* mesh = geom_cache->scene->objects[object]->mesh;
	mesh->subpatches[prim].cached_tessellated_size = size;
}

TessellatedSubPatch* geom_cache_get_subpatch(KernelGlobals *kg, int object, int prim)
{
	GeomCache* geom_cache = kg->geom_cache;
	GeomCacheThreadData* tdata = kg->geom_cache_tdata;

	GeomCache::lru_t* lru = &geom_cache->lru;

	GeomCache::patch_key_t key(object, prim);
	GeomCache::lru_t::ref_t ref;

	if(!lru->find_or_lock(key, ref, tdata->lru_tdata)) {
		// get patch size
		uint num_verts, num_tris, bvh_size = 0;
		int total_size = -1;
		geom_cache_get_tessellated_subpatch_size(geom_cache, object, prim, &num_verts, &num_tris, &total_size);
		bvh_size = num_tris*2*4 + 4;

		size_t size = sizeof(SubPatchWraper) + sizeof(float4)*(num_verts*2 + num_tris + bvh_size);
		if(total_size >= 0)
			size = total_size;

		// alloc
		SubPatchWraper* wraper = (SubPatchWraper*)operator new (size);
		memset(wraper, 0, sizeof(SubPatchWraper));
		TessellatedSubPatch* subpatch = &wraper->subpatch;

		// insert asap to unblock other threads
		ref = wraper;
		ref.set_size(size);
		lru->insert_and_unlock(key, ref, tdata->lru_tdata);

		subpatch->object = object;
		subpatch->prim = prim;

		subpatch->vert_offset = 0;
		subpatch->tri_offset = num_verts*2;
		subpatch->bvh_offset = subpatch->tri_offset + num_tris;

		// dice
		geom_cache_dice_subpatch(geom_cache, subpatch, object, prim);

		// displace
		// TODO(mai): implement

		// build bvh
		bvh_size = subpatch_build_bvh(subpatch, bvh_size);

		// update size for next time
		if(total_size < 0) {
			size = sizeof(TessellatedSubPatch) + sizeof(float4)*(num_verts*2 + subpatch->num_triangles + bvh_size);
			geom_cache_update_subpatch_size(geom_cache, object, prim, size);
		}

		// signal subpatch completion
		ref.mark_ready();
	}
	else {
		ref.wait_till_ready();
	}
	
	/* extra ref so subpatch doesnt deallocate when falling out of scope here, caller needs a ref without ref_t */
	ref.inc();
	return &ref->subpatch;
}

void geom_cache_release_subpatch(KernelGlobals *kg, TessellatedSubPatch *subpatch) {
	SubPatchWraper* wraper = (SubPatchWraper*)((char*)subpatch - offsetof(SubPatchWraper, subpatch));

	GeomCache::lru_t::ref_t ref(wraper);
	ref.dec(); /* extra unref to cause subpatch to dellocated when scope ends (if ref not held elsewhere) */
}

CCL_NAMESPACE_END


