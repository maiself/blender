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

CCL_NAMESPACE_BEGIN

struct GeomCache;

GeomCache* geom_cache_create();
void geom_cache_release(GeomCache *geom_cache);

void geom_cache_set_scene(GeomCache *geom_cache, void *scene);
void geom_cache_thread_init(KernelGlobals *kg, GeomCache *geom_cache);
void geom_cache_clear(GeomCache *geom_cache);
void geom_cache_set_max_size(GeomCache * geom_cache, size_t max_size);

TessellatedSubPatch* geom_cache_get_subpatch(KernelGlobals *kg, int object, int prim);
void geom_cache_release_subpatch(KernelGlobals *kg, TessellatedSubPatch *subpatch);

void geom_cache_sample_subpatch_vert(KernelGlobals *kg, int object, int prim, int vert, float3 *P, float3 *N,
                                     float *u, float *v, int *shader);
void geom_cache_sample_subpatch_vert_displacement(KernelGlobals *kg, int object, int prim, int vert, float3 *dP);

CCL_NAMESPACE_END


