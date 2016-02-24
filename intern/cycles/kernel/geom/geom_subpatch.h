/*
 * Copyright 2016, Blender Foundation.
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

ccl_device_inline bool subpatch_is_quad(TessellatedSubPatch* patch) {
	return !(patch->uv[3].x == -1);
}

/* Workaround stupidness of CUDA/OpenCL which doesn't allow to access indexed
 * component of float3 value.
 */
#ifndef __KERNEL_CPU__
#  define IDX(vec, idx) \
    ((idx == 0) ? ((vec).x) : ( (idx == 1) ? ((vec).y) : ((vec).z) ))
#else
#  define IDX(vec, idx) ((vec)[idx])
#endif

ccl_device_inline bool subpatch_triangle_intersect(KernelGlobals *kg,
                                                   const IsectPrecalc *isect_precalc,
                                                   Intersection *isect,
                                                   float3 P,
                                                   const float4 tri_a,
                                                   const float4 tri_b,
                                                   const float4 tri_c)
{
	const int kx = isect_precalc->kx;
	const int ky = isect_precalc->ky;
	const int kz = isect_precalc->kz;
	const float Sx = isect_precalc->Sx;
	const float Sy = isect_precalc->Sy;
	const float Sz = isect_precalc->Sz;

	/* Calculate vertices relative to ray origin. */
	const float3 A = make_float3(tri_a.x - P.x, tri_a.y - P.y, tri_a.z - P.z);
	const float3 B = make_float3(tri_b.x - P.x, tri_b.y - P.y, tri_b.z - P.z);
	const float3 C = make_float3(tri_c.x - P.x, tri_c.y - P.y, tri_c.z - P.z);

	const float A_kx = IDX(A, kx), A_ky = IDX(A, ky), A_kz = IDX(A, kz);
	const float B_kx = IDX(B, kx), B_ky = IDX(B, ky), B_kz = IDX(B, kz);
	const float C_kx = IDX(C, kx), C_ky = IDX(C, ky), C_kz = IDX(C, kz);

	/* Perform shear and scale of vertices. */
	const float Ax = A_kx - Sx * A_kz;
	const float Ay = A_ky - Sy * A_kz;
	const float Bx = B_kx - Sx * B_kz;
	const float By = B_ky - Sy * B_kz;
	const float Cx = C_kx - Sx * C_kz;
	const float Cy = C_ky - Sy * C_kz;

	/* Calculate scaled barycentric coordinates. */
	float U = Cx * By - Cy * Bx;
	float V = Ax * Cy - Ay * Cx;
	float W = Bx * Ay - By * Ax;
	if((U < 0.0f || V < 0.0f || W < 0.0f) &&
	   (U > 0.0f || V > 0.0f || W > 0.0f))
	{
		return false;
	}

	/* Calculate determinant. */
	float det = U + V + W;
	if(UNLIKELY(det == 0.0f)) {
		return false;
	}

	/* Calculate scaled z-coordinates of vertices and use them to calculate
	 * the hit distance.
	 */
	const float T = (U * A_kz + V * B_kz + W * C_kz) * Sz;
	const int sign_det = (__float_as_int(det) & 0x80000000);
	const float sign_T = xor_signmask(T, sign_det);
	if((sign_T < 0.0f) ||
	   (sign_T > isect->t * xor_signmask(det, sign_det)))
	{
		return false;
	}

	/* Normalize U, V, W, and T. */
	const float inv_det = 1.0f / det;
	isect->u = U * inv_det;
	isect->v = V * inv_det;
	isect->t = T * inv_det;
	return true;
}

/* Special ray intersection routines for subsurface scattering. In that case we
 * only want to intersect with primitives in the same object, and if case of
 * multiple hits we pick a single random primitive as the intersection point.
 */

#ifdef __SUBSURFACE__
ccl_device_inline Intersection* subpatch_triangle_intersect_subsurface(
        KernelGlobals *kg,
        const IsectPrecalc *isect_precalc,
        SubsurfaceIntersection *ss_isect,
        float3 P,
        float tmax,
        uint *lcg_state,
        int max_hits,
        const float4 tri_a,
        const float4 tri_b,
        const float4 tri_c)
{
	const int kx = isect_precalc->kx;
	const int ky = isect_precalc->ky;
	const int kz = isect_precalc->kz;
	const float Sx = isect_precalc->Sx;
	const float Sy = isect_precalc->Sy;
	const float Sz = isect_precalc->Sz;

	/* Calculate vertices relative to ray origin. */
	const float3 A = make_float3(tri_a.x - P.x, tri_a.y - P.y, tri_a.z - P.z);
	const float3 B = make_float3(tri_b.x - P.x, tri_b.y - P.y, tri_b.z - P.z);
	const float3 C = make_float3(tri_c.x - P.x, tri_c.y - P.y, tri_c.z - P.z);

	const float A_kx = IDX(A, kx), A_ky = IDX(A, ky), A_kz = IDX(A, kz);
	const float B_kx = IDX(B, kx), B_ky = IDX(B, ky), B_kz = IDX(B, kz);
	const float C_kx = IDX(C, kx), C_ky = IDX(C, ky), C_kz = IDX(C, kz);

	/* Perform shear and scale of vertices. */
	const float Ax = A_kx - Sx * A_kz;
	const float Ay = A_ky - Sy * A_kz;
	const float Bx = B_kx - Sx * B_kz;
	const float By = B_ky - Sy * B_kz;
	const float Cx = C_kx - Sx * C_kz;
	const float Cy = C_ky - Sy * C_kz;

	/* Calculate scaled barycentric coordinates. */
	float U = Cx * By - Cy * Bx;
	float V = Ax * Cy - Ay * Cx;
	float W = Bx * Ay - By * Ax;

	if((U < 0.0f || V < 0.0f || W < 0.0f) &&
	   (U > 0.0f || V > 0.0f || W > 0.0f))
	{
		return NULL;
	}

	/* Calculate determinant. */
	float det = U + V + W;
	if(UNLIKELY(det == 0.0f)) {
		return NULL;
	}

	/* Calculate scaled z-coordinates of vertices and use them to calculate
	 * the hit distance.
	 */
	const int sign_det = (__float_as_int(det) & 0x80000000);
	const float T = (U * A_kz + V * B_kz + W * C_kz) * Sz;
	const float sign_T = xor_signmask(T, sign_det);
	if((sign_T < 0.0f) ||
	   (sign_T > tmax * xor_signmask(det, sign_det)))
	{
		return NULL;
	}

	/* Normalize U, V, W, and T. */
	const float inv_det = 1.0f / det;

	ss_isect->num_hits++;
	int hit;

	if(ss_isect->num_hits <= max_hits) {
		hit = ss_isect->num_hits - 1;
	}
	else {
		/* reservoir sampling: if we are at the maximum number of
		 * hits, randomly replace element or skip it */
		hit = lcg_step_uint(lcg_state) % ss_isect->num_hits;

		if(hit >= max_hits)
			return NULL;
	}

	/* record intersection */
	Intersection *isect = &ss_isect->hits[hit];
	isect->u = U * inv_det;
	isect->v = V * inv_det;
	isect->t = T * inv_det;

	/* Record geometric normal. */
	/* TODO(sergey): Use float4_to_float3() on just an edges. */
	const float3 v0 = float4_to_float3(tri_a);
	const float3 v1 = float4_to_float3(tri_b);
	const float3 v2 = float4_to_float3(tri_c);
	ss_isect->Ng[hit] = normalize(cross(v1 - v0, v2 - v0));

	return isect;
}
#endif

#undef IDX

ccl_device_inline void subpatch_intersect_store_cache_triangle(Intersection *isect,
		TessellatedSubPatch* subpatch, float4* v, float4* n)
{
	CacheTriangle tri;

	for(int i = 0; i < 3; i++) {
		tri.verts[i] = float4_to_float3(v[i]);
		tri.normals[i] = normalize(float4_to_float3(n[i]));

		/* uv within subpatch */
		float u_ = v[i].w;
		float v_ = n[i].w;

		/* uv within patch */
		tri.uv[i] = make_float2(u_, v_);
#if 0
		if(subpatch_is_quad(subpatch)) {
			tri.uv[i] = interp(interp(subpatch->uv[0], subpatch->uv[1], u_),
			                   interp(subpatch->uv[2], subpatch->uv[3], u_), v_);
		}
		else {
			tri.uv[i] = u_*subpatch->uv[0] + v_*subpatch->uv[1] + (1.0f-u_-v_)*subpatch->uv[2];
		}
#endif
	}

	for(int i = 0; i < 4; i++)
		tri.v[i] = subpatch->v[i];

	tri.patch = subpatch->patch;
	tri.shader = subpatch->shader;

	isect->cache_triangle = tri;
}

ccl_device_inline bool subpatch_intersect(KernelGlobals *kg,
                                          const IsectPrecalc *isect_precalc,
                                          Intersection *isect,
                                          float3 P,
                                          uint visibility,
                                          int object,
                                          int patchAddr)
{
#if 0 /*__VISIBILITY_FLAG__*/
	if(kernel_tex_fetch(__prim_visibility, patchAddr) & visibility)
		return false;
#endif

	/* get subpatch from cache */
	TessellatedSubPatch* subpatch = geom_cache_get_subpatch(kg, object, kernel_tex_fetch(__prim_index, patchAddr));

	float4* verts = &subpatch->data[subpatch->vert_offset];
	uint4* indices = (uint4*)&subpatch->data[subpatch->tri_offset];

	float4 v[3];
	float4 n[3];
	bool hit = false;

	/* TODO(mai): implement bvh for subpatches */
	/* test triangles */
	for(int i = 0; i < subpatch->num_triangles; i++) {
		if(subpatch_triangle_intersect(kg, isect_precalc, isect, P,
				verts[indices[i].x], verts[indices[i].y], verts[indices[i].z]))
		{
			hit = true;

			v[0] = verts[indices[i].x];
			v[1] = verts[indices[i].y];
			v[2] = verts[indices[i].z];

			n[0] = verts[indices[i].x+1];
			n[1] = verts[indices[i].y+1];
			n[2] = verts[indices[i].z+1];

			isect->prim = patchAddr;
			isect->object = object;
			isect->type = PRIMITIVE_CACHE_TRIANGLE;

			/* shadow ray early termination */
			if(visibility == PATH_RAY_SHADOW_OPAQUE)
				break;
		}
	}

	if(hit) {
		/* store cached triangle since subpatch data may be invalid after leaving this function */
		subpatch_intersect_store_cache_triangle(isect, subpatch, v, n);
	}

	geom_cache_release_subpatch(kg, subpatch);
	return hit;
}

ccl_device_inline bool subpatch_intersect_shadow(KernelGlobals *kg,
                                                 const IsectPrecalc *isect_precalc,
                                                 Intersection **isect_array,
                                                 const uint max_hits,
                                                 uint *num_hits,
                                                 float isect_t,
                                                 float3 P,
                                                 int object,
                                                 int patchAddr)
{
	/* get subpatch from cache */
	TessellatedSubPatch* subpatch = geom_cache_get_subpatch(kg, object, kernel_tex_fetch(__prim_index, patchAddr));

	float4* verts = &subpatch->data[subpatch->vert_offset];
	uint4* indices = (uint4*)&subpatch->data[subpatch->tri_offset];

	float4 v[3];
	float4 n[3];
	bool hit = false;

	/* TODO(mai): implement bvh for subpatches */
	/* test triangles */
	for(int i = 0; i < subpatch->num_triangles; i++) {
		if(subpatch_triangle_intersect(kg, isect_precalc, *isect_array, P,
				verts[indices[i].x], verts[indices[i].y], verts[indices[i].z]))
		{
			v[0] = verts[indices[i].x];
			v[1] = verts[indices[i].y];
			v[2] = verts[indices[i].z];

			n[0] = verts[indices[i].x+1];
			n[1] = verts[indices[i].y+1];
			n[2] = verts[indices[i].z+1];

			(*isect_array)->prim = patchAddr;
			(*isect_array)->object = object;
			(*isect_array)->type = PRIMITIVE_CACHE_TRIANGLE;

			/* store cached triangle since subpatch data may be invalid after leaving this function */
			subpatch_intersect_store_cache_triangle(*isect_array, subpatch, v, n);

			/* shadow ray early termination */

			/* detect if this surface has a shader with transparent shadows */
			int flag = kernel_tex_fetch(__shader_flag, (subpatch->shader & SHADER_MASK)*2);

			/* if no transparent shadows, all light is blocked */
			if(!(flag & SD_HAS_TRANSPARENT_SHADOW)) {
				hit = true;
				break;
			}
			/* if maximum number of hits reached, block all light */
			else if(*num_hits == max_hits) {
				hit = true;
				break;
			}

			/* move on to next entry in intersections array */
			(*isect_array)++;
			(*num_hits)++;

			(*isect_array)->t = isect_t;
		}
	}

	geom_cache_release_subpatch(kg, subpatch);
	return hit;
}

ccl_device_inline bool subpatch_intersect_volume_all(KernelGlobals *kg,
                                                     const IsectPrecalc *isect_precalc,
                                                     Intersection **isect_array,
                                                     const uint max_hits,
                                                     uint *num_hits,
                                                     int *num_hits_in_instance,
                                                     float isect_t,
                                                     float3 P,
                                                     float3 dir,
                                                     uint visibility,
                                                     Transform* itfm,
                                                     int object,
                                                     int patchAddr)
{
#if 0 /*__VISIBILITY_FLAG__*/
	if(kernel_tex_fetch(__prim_visibility, patchAddr) & visibility)
		return false;
#endif

	/* get subpatch from cache */
	TessellatedSubPatch* subpatch = geom_cache_get_subpatch(kg, object, kernel_tex_fetch(__prim_index, patchAddr));

	float4* verts = &subpatch->data[subpatch->vert_offset];
	uint4* indices = (uint4*)&subpatch->data[subpatch->tri_offset];

	float4 v[3];
	float4 n[3];
	bool hit = false;

	/* TODO(mai): implement bvh for subpatches */
	/* test triangles */
	for(int i = 0; i < subpatch->num_triangles; i++) {
		if(subpatch_triangle_intersect(kg, isect_precalc, *isect_array, P,
				verts[indices[i].x], verts[indices[i].y], verts[indices[i].z]))
		{
			v[0] = verts[indices[i].x];
			v[1] = verts[indices[i].y];
			v[2] = verts[indices[i].z];

			n[0] = verts[indices[i].x+1];
			n[1] = verts[indices[i].y+1];
			n[2] = verts[indices[i].z+1];

			(*isect_array)->prim = patchAddr;
			(*isect_array)->object = object;
			(*isect_array)->type = PRIMITIVE_CACHE_TRIANGLE;

			/* store cached triangle since subpatch data may be invalid after leaving this function */
			subpatch_intersect_store_cache_triangle(*isect_array, subpatch, v, n);

			/* move on to next entry in intersections array */
			(*isect_array)++;
			(*num_hits)++;

			if(num_hits_in_instance) {
				(*num_hits_in_instance)++;
			}

			(*isect_array)->t = isect_t;

			if(*num_hits == max_hits) {
				if(num_hits_in_instance) {
					float t_fac = 1.0f / len(transform_direction(itfm, dir));

					for(int i = 0; i < *num_hits_in_instance; i++) {
						((*isect_array)-i-1)->t *= t_fac;
					}
				}

				hit = true;
				break;
			}
		}
	}

	geom_cache_release_subpatch(kg, subpatch);
	return hit;
}

#ifdef __SUBSURFACE__
ccl_device_inline void subpatch_intersect_subsurface(
        KernelGlobals *kg,
        const IsectPrecalc *isect_precalc,
        SubsurfaceIntersection *ss_isect,
        float3 P,
        int object,
        int patchAddr,
        float tmax,
        uint *lcg_state,
        int max_hits)
{
#if 0 /*__VISIBILITY_FLAG__*/
	if(kernel_tex_fetch(__prim_visibility, patchAddr) & visibility)
		return false;
#endif

	/* get subpatch from cache */
	TessellatedSubPatch* subpatch = geom_cache_get_subpatch(kg, object, kernel_tex_fetch(__prim_index, patchAddr));

	float4* verts = &subpatch->data[subpatch->vert_offset];
	uint4* indices = (uint4*)&subpatch->data[subpatch->tri_offset];

	float4 v[3];
	float4 n[3];
	Intersection *isect;

	/* TODO(mai): implement bvh for subpatches */
	/* test triangles */
	for(int i = 0; i < subpatch->num_triangles; i++) {
		if(( isect = subpatch_triangle_intersect_subsurface(kg, isect_precalc, ss_isect, P, tmax, lcg_state, max_hits,
				verts[indices[i].x], verts[indices[i].y], verts[indices[i].z]) ))
		{
			v[0] = verts[indices[i].x];
			v[1] = verts[indices[i].y];
			v[2] = verts[indices[i].z];

			n[0] = verts[indices[i].x+1];
			n[1] = verts[indices[i].y+1];
			n[2] = verts[indices[i].z+1];

			isect->prim = patchAddr;
			isect->object = object;
			isect->type = PRIMITIVE_CACHE_TRIANGLE;

			/* store cached triangle since subpatch data may be invalid after leaving this function */
			subpatch_intersect_store_cache_triangle(isect, subpatch, v, n);
		}
	}

	geom_cache_release_subpatch(kg, subpatch);
}
#endif

CCL_NAMESPACE_END


