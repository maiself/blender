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

ccl_device_inline bool cache_triangle_patch_is_quad(KernelGlobals *kg, const ShaderData *sd) {
	return !(ccl_fetch(sd, cache_triangle).patch & 0x80000000);
}

ccl_device_inline float3 cache_triangle_refine(KernelGlobals *kg,
                                         ShaderData *sd,
                                         const Intersection *isect,
                                         const Ray *ray)
{
	float3 P = ray->P;
	float3 D = ray->D;
	float t = isect->t;

#ifdef __INTERSECTION_REFINE__
	if(isect->object != OBJECT_NONE) {
		if(UNLIKELY(t == 0.0f)) {
			return P;
		}
#ifdef __OBJECT_MOTION__
		Transform tfm = ccl_fetch(sd, ob_itfm);
#else
		Transform tfm = object_fetch_transform(kg, isect->object, OBJECT_INVERSE_TRANSFORM);
#endif

		P = transform_point(&tfm, P);
		D = transform_direction(&tfm, D*t);
		D = normalize_len(D, &t);
	}

	P = P + D*t;

	const float3 tri_a = sd->cache_triangle.verts[0],
	             tri_b = sd->cache_triangle.verts[1],
	             tri_c = sd->cache_triangle.verts[2];
	float3 edge1 = make_float3(tri_a.x - tri_c.x, tri_a.y - tri_c.y, tri_a.z - tri_c.z);
	float3 edge2 = make_float3(tri_b.x - tri_c.x, tri_b.y - tri_c.y, tri_b.z - tri_c.z);
	float3 tvec = make_float3(P.x - tri_c.x, P.y - tri_c.y, P.z - tri_c.z);
	float3 qvec = cross(tvec, edge1);
	float3 pvec = cross(D, edge2);
	float rt = dot(edge2, qvec) / dot(edge1, pvec);

	P = P + D*rt;

	if(isect->object != OBJECT_NONE) {
#ifdef __OBJECT_MOTION__
		Transform tfm = ccl_fetch(sd, ob_tfm);
#else
		Transform tfm = object_fetch_transform(kg, isect->object, OBJECT_TRANSFORM);
#endif

		P = transform_point(&tfm, P);
	}

	return P;
#else
	return P + D*t;
#endif
}

/* Same as above, except that isect->t is assumed to be in object space for
 * instancing.
 */
ccl_device_inline float3 cache_triangle_refine_subsurface(KernelGlobals *kg,
                                                    ShaderData *sd,
                                                    CacheTriangle* cache_triangle,
                                                    const Intersection *isect,
                                                    const Ray *ray)
{
	float3 P = ray->P;
	float3 D = ray->D;
	float t = isect->t;

	if(isect->object != OBJECT_NONE) {
#ifdef __OBJECT_MOTION__
		Transform tfm = ccl_fetch(sd, ob_itfm);
#else
		Transform tfm = object_fetch_transform(kg,
		                                       isect->object,
		                                       OBJECT_INVERSE_TRANSFORM);
#endif

		P = transform_point(&tfm, P);
		D = transform_direction(&tfm, D);
		D = normalize(D);
	}

	P = P + D*t;

#ifdef __INTERSECTION_REFINE__
	const float3 tri_a = cache_triangle->verts[0],
	             tri_b = cache_triangle->verts[1],
	             tri_c = cache_triangle->verts[2];
	float3 edge1 = make_float3(tri_a.x - tri_c.x, tri_a.y - tri_c.y, tri_a.z - tri_c.z);
	float3 edge2 = make_float3(tri_b.x - tri_c.x, tri_b.y - tri_c.y, tri_b.z - tri_c.z);
	float3 tvec = make_float3(P.x - tri_c.x, P.y - tri_c.y, P.z - tri_c.z);
	float3 qvec = cross(tvec, edge1);
	float3 pvec = cross(D, edge2);
	float rt = dot(edge2, qvec) / dot(edge1, pvec);

	P = P + D*rt;
#endif  /* __INTERSECTION_REFINE__ */

	if(isect->object != OBJECT_NONE) {
#ifdef __OBJECT_MOTION__
		Transform tfm = ccl_fetch(sd, ob_tfm);
#else
		Transform tfm = object_fetch_transform(kg,
		                                       isect->object,
		                                       OBJECT_TRANSFORM);
#endif

		P = transform_point(&tfm, P);
	}

	return P;
}

ccl_device_inline float3 cache_triangle_smooth_normal(KernelGlobals *kg, CacheTriangle *tri, float u, float v)
{
	return normalize(u*tri->normals[0] + v*tri->normals[1] + (1.0f-u-v)*tri->normals[2]);
}

ccl_device_noinline void cache_triangle_shader_setup(KernelGlobals *kg, ShaderData *sd, const Intersection *isect, const Ray *ray, bool subsurface)
{
	ccl_fetch(sd, cache_triangle) = isect->cache_triangle;
	CacheTriangle* tri = &ccl_fetch(sd, cache_triangle);

	ccl_fetch(sd, shader) = tri->shader;

	/* compute refined position */
#ifdef __SUBSURFACE__
	if(!subsurface)
#endif
		ccl_fetch(sd, P) = cache_triangle_refine(kg, sd, isect, ray);
#ifdef __SUBSURFACE__
	else
		ccl_fetch(sd, P) = cache_triangle_refine_subsurface(kg, sd, &sd->cache_triangle, isect, ray);
#endif

	/* compute face normal */
	float3 Ng;
	if(ccl_fetch(sd, flag) & SD_NEGATIVE_SCALE_APPLIED)
		Ng = normalize(cross(tri->verts[2] - tri->verts[0], tri->verts[1] - tri->verts[0]));
	else
		Ng = normalize(cross(tri->verts[1] - tri->verts[0], tri->verts[2] - tri->verts[0]));

	ccl_fetch(sd, Ng) = Ng;
	if(ccl_fetch(sd, shader) & SHADER_SMOOTH_NORMAL)
		ccl_fetch(sd, N) = cache_triangle_smooth_normal(kg, tri, isect->u, isect->v);
	else
		ccl_fetch(sd, N) = Ng;

	/* compute derivatives of P w.r.t. uv */
#ifdef __DPDU__
	ccl_fetch(sd, dPdu) = (tri->verts[0] - tri->verts[2]);
	ccl_fetch(sd, dPdv) = (tri->verts[1] - tri->verts[2]);
#endif
}

ccl_device float cache_triangle_attribute_float(KernelGlobals *kg, const ShaderData *sd, AttributeElement elem, int offset, float *dx, float *dy)
{
	if(elem == ATTR_ELEMENT_FACE) {
		if(dx) *dx = 0.0f;
		if(dy) *dy = 0.0f;

		return kernel_tex_fetch(__attributes_float, offset + ccl_fetch(sd, prim));
	}
	else if(elem == ATTR_ELEMENT_VERTEX || elem == ATTR_ELEMENT_VERTEX_MOTION) {
		float2 uv[3];
		for(int i = 0; i < 3; i++) {
			uv[i] = ccl_fetch(sd, cache_triangle).uv[i];
		}

		float a, b, c;

		float f0 = kernel_tex_fetch(__attributes_float, offset + ccl_fetch(sd, cache_triangle).v[0]);
		float f1 = kernel_tex_fetch(__attributes_float, offset + ccl_fetch(sd, cache_triangle).v[1]);
		float f2 = kernel_tex_fetch(__attributes_float, offset + ccl_fetch(sd, cache_triangle).v[2]);

		if(cache_triangle_patch_is_quad(kg, sd)) {
			float f3 = kernel_tex_fetch(__attributes_float, offset + ccl_fetch(sd, cache_triangle).v[3]);

			a = interp(interp(f0, f1, uv[0].x), interp(f3, f2, uv[0].x), uv[0].y);
			b = interp(interp(f0, f1, uv[1].x), interp(f3, f2, uv[1].x), uv[1].y);
			c = interp(interp(f0, f1, uv[2].x), interp(f3, f2, uv[2].x), uv[2].y);
		}
		else {
			a = uv[0].x*f0 + uv[0].y*f1 + (1.0f - uv[0].x - uv[0].y)*f2;
			b = uv[1].x*f0 + uv[1].y*f1 + (1.0f - uv[1].x - uv[1].y)*f2;
			c = uv[2].x*f0 + uv[2].y*f1 + (1.0f - uv[2].x - uv[2].y)*f2;
		}

#ifdef __RAY_DIFFERENTIALS__
		if(dx) *dx = ccl_fetch(sd, du).dx*a + ccl_fetch(sd, dv).dx*b - (ccl_fetch(sd, du).dx + ccl_fetch(sd, dv).dx)*c;
		if(dy) *dy = ccl_fetch(sd, du).dy*a + ccl_fetch(sd, dv).dy*b - (ccl_fetch(sd, du).dy + ccl_fetch(sd, dv).dy)*c;
#endif

		return ccl_fetch(sd, u)*a + ccl_fetch(sd, v)*b + (1.0f - ccl_fetch(sd, u) - ccl_fetch(sd, v))*c;
	}
	else if(elem == ATTR_ELEMENT_CORNER) {
		int patch = offset + (ccl_fetch(sd, cache_triangle).patch & 0x7fffffff)*4;

		float2 uv[3];
		for(int i = 0; i < 3; i++) {
			uv[i] = ccl_fetch(sd, cache_triangle).uv[i];
		}

		float a, b, c;

		float f0 = kernel_tex_fetch(__attributes_float, patch + 0);
		float f1 = kernel_tex_fetch(__attributes_float, patch + 1);
		float f2 = kernel_tex_fetch(__attributes_float, patch + 2);

		if(cache_triangle_patch_is_quad(kg, sd)) {
			float f3 = kernel_tex_fetch(__attributes_float, patch + 3);

			a = interp(interp(f0, f1, uv[0].x), interp(f3, f2, uv[0].x), uv[0].y);
			b = interp(interp(f0, f1, uv[1].x), interp(f3, f2, uv[1].x), uv[1].y);
			c = interp(interp(f0, f1, uv[2].x), interp(f3, f2, uv[2].x), uv[2].y);
		}
		else {
			a = uv[0].x*f0 + uv[0].y*f1 + (1.0f - uv[0].x - uv[0].y)*f2;
			b = uv[1].x*f0 + uv[1].y*f1 + (1.0f - uv[1].x - uv[1].y)*f2;
			c = uv[2].x*f0 + uv[2].y*f1 + (1.0f - uv[2].x - uv[2].y)*f2;
		}

#ifdef __RAY_DIFFERENTIALS__
		if(dx) *dx = ccl_fetch(sd, du).dx*a + ccl_fetch(sd, dv).dx*b - (ccl_fetch(sd, du).dx + ccl_fetch(sd, dv).dx)*c;
		if(dy) *dy = ccl_fetch(sd, du).dy*a + ccl_fetch(sd, dv).dy*b - (ccl_fetch(sd, du).dy + ccl_fetch(sd, dv).dy)*c;
#endif

		return ccl_fetch(sd, u)*a + ccl_fetch(sd, v)*b + (1.0f - ccl_fetch(sd, u) - ccl_fetch(sd, v))*c;
	}
	else {
		if(dx) *dx = 0.0f;
		if(dy) *dy = 0.0f;

		return 0.0f;
	}
}

ccl_device float3 cache_triangle_attribute_float3(KernelGlobals *kg, const ShaderData *sd, AttributeElement elem, int offset, float3 *dx, float3 *dy)
{
	if(elem == ATTR_ELEMENT_FACE) {
		if(dx) *dx = make_float3(0.0f, 0.0f, 0.0f);
		if(dy) *dy = make_float3(0.0f, 0.0f, 0.0f);

		return float4_to_float3(kernel_tex_fetch(__attributes_float3, offset + ccl_fetch(sd, prim)));
	}
	else if(elem == ATTR_ELEMENT_VERTEX || elem == ATTR_ELEMENT_VERTEX_MOTION) {
		float2 uv[3];
		for(int i = 0; i < 3; i++) {
			uv[i] = ccl_fetch(sd, cache_triangle).uv[i];
		}

		float3 a, b, c;

		float3 f0 = float4_to_float3(kernel_tex_fetch(__attributes_float3, offset + ccl_fetch(sd, cache_triangle).v[0]));
		float3 f1 = float4_to_float3(kernel_tex_fetch(__attributes_float3, offset + ccl_fetch(sd, cache_triangle).v[1]));
		float3 f2 = float4_to_float3(kernel_tex_fetch(__attributes_float3, offset + ccl_fetch(sd, cache_triangle).v[2]));

		if(cache_triangle_patch_is_quad(kg, sd)) {
			float3 f3 = float4_to_float3(kernel_tex_fetch(__attributes_float3, offset + ccl_fetch(sd, cache_triangle).v[3]));

			a = interp(interp(f0, f1, uv[0].x), interp(f3, f2, uv[0].x), uv[0].y);
			b = interp(interp(f0, f1, uv[1].x), interp(f3, f2, uv[1].x), uv[1].y);
			c = interp(interp(f0, f1, uv[2].x), interp(f3, f2, uv[2].x), uv[2].y);
		}
		else {
			a = uv[0].x*f0 + uv[0].y*f1 + (1.0f - uv[0].x - uv[0].y)*f2;
			b = uv[1].x*f0 + uv[1].y*f1 + (1.0f - uv[1].x - uv[1].y)*f2;
			c = uv[2].x*f0 + uv[2].y*f1 + (1.0f - uv[2].x - uv[2].y)*f2;
		}

#ifdef __RAY_DIFFERENTIALS__
		if(dx) *dx = ccl_fetch(sd, du).dx*a + ccl_fetch(sd, dv).dx*b - (ccl_fetch(sd, du).dx + ccl_fetch(sd, dv).dx)*c;
		if(dy) *dy = ccl_fetch(sd, du).dy*a + ccl_fetch(sd, dv).dy*b - (ccl_fetch(sd, du).dy + ccl_fetch(sd, dv).dy)*c;
#endif

		return ccl_fetch(sd, u)*a + ccl_fetch(sd, v)*b + (1.0f - ccl_fetch(sd, u) - ccl_fetch(sd, v))*c;
	}
	else if(elem == ATTR_ELEMENT_CORNER || elem == ATTR_ELEMENT_CORNER_BYTE) {
		int patch = offset + (ccl_fetch(sd, cache_triangle).patch & 0x7fffffff)*4;

		float2 uv[3];
		for(int i = 0; i < 3; i++) {
			uv[i] = ccl_fetch(sd, cache_triangle).uv[i];
		}

		float3 a, b, c;
		float3 f0, f1, f2, f3;

		if(elem == ATTR_ELEMENT_CORNER) {
			f0 = float4_to_float3(kernel_tex_fetch(__attributes_float3, patch + 0));
			f1 = float4_to_float3(kernel_tex_fetch(__attributes_float3, patch + 1));
			f2 = float4_to_float3(kernel_tex_fetch(__attributes_float3, patch + 2));
		}
		else {
			f0 = color_byte_to_float(kernel_tex_fetch(__attributes_uchar4, patch + 0));
			f1 = color_byte_to_float(kernel_tex_fetch(__attributes_uchar4, patch + 1));
			f2 = color_byte_to_float(kernel_tex_fetch(__attributes_uchar4, patch + 2));
		}

		if(cache_triangle_patch_is_quad(kg, sd)) {
			if(elem == ATTR_ELEMENT_CORNER) {
				f3 = float4_to_float3(kernel_tex_fetch(__attributes_float3, patch + 3));
			}
			else {
				f3 = color_byte_to_float(kernel_tex_fetch(__attributes_uchar4, patch + 3));
			}

			a = interp(interp(f0, f1, uv[0].x), interp(f3, f2, uv[0].x), uv[0].y);
			b = interp(interp(f0, f1, uv[1].x), interp(f3, f2, uv[1].x), uv[1].y);
			c = interp(interp(f0, f1, uv[2].x), interp(f3, f2, uv[2].x), uv[2].y);
		}
		else {
			a = uv[0].x*f0 + uv[0].y*f1 + (1.0f - uv[0].x - uv[0].y)*f2;
			b = uv[1].x*f0 + uv[1].y*f1 + (1.0f - uv[1].x - uv[1].y)*f2;
			c = uv[2].x*f0 + uv[2].y*f1 + (1.0f - uv[2].x - uv[2].y)*f2;
		}

#ifdef __RAY_DIFFERENTIALS__
		if(dx) *dx = ccl_fetch(sd, du).dx*a + ccl_fetch(sd, dv).dx*b - (ccl_fetch(sd, du).dx + ccl_fetch(sd, dv).dx)*c;
		if(dy) *dy = ccl_fetch(sd, du).dy*a + ccl_fetch(sd, dv).dy*b - (ccl_fetch(sd, du).dy + ccl_fetch(sd, dv).dy)*c;
#endif

		return ccl_fetch(sd, u)*a + ccl_fetch(sd, v)*b + (1.0f - ccl_fetch(sd, u) - ccl_fetch(sd, v))*c;
	}
	else {
		if(dx) *dx = make_float3(0.0f, 0.0f, 0.0f);;
		if(dy) *dy = make_float3(0.0f, 0.0f, 0.0f);;

		return make_float3(0.0f, 0.0f, 0.0f);;
	}
}

CCL_NAMESPACE_END

