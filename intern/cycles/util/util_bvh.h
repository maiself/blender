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

#ifndef __UTIL_BVH_H__
#define __UTIL_BVH_H__

#include "util_math.h"

CCL_NAMESPACE_BEGIN

/* builds a bvh from a triangle soup 
 * 
 * size of bvh_data must be at least (sizeof(float4) * num_triangles * 8 + 4) or
 * actual size of built bvh if known ahead of time
 *
 * returns number of float4 used for bvh
 */
size_t util_bvh_build(float4* bvh_data, float4* verts, uint4* triangles, uint num_triangles, uint size);

CCL_NAMESPACE_END

#endif /* __UTIL_BVH_H__ */

