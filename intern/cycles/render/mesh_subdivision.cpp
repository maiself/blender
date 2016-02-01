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

#include "bvh.h"
#include "bvh_build.h"

#include "camera.h"
#include "curves.h"
#include "device.h"
#include "graph.h"
#include "shader.h"
#include "light.h"
#include "mesh.h"
#include "nodes.h"
#include "object.h"
#include "scene.h"

#include "osl_globals.h"

#include "util_foreach.h"
#include "util_logging.h"
#include "util_progress.h"
#include "util_set.h"

#include "../subd/subd_split.h"
#include "../subd/subd_patch.h"

CCL_NAMESPACE_BEGIN

static float3 patch_normal(Mesh* mesh, int patch) {
	Mesh::Patch& t = mesh->patches[patch];

	float3 v0 = mesh->verts[t.v[0]];
	float3 v1 = mesh->verts[t.v[1]];
	float3 v2 = mesh->verts[t.v[2]];

	float3 norm = cross(v1 - v0, v2 - v0);
	float normlen = len(norm);

	if(normlen == 0.0f)
		return make_float3(0.0f, 0.0f, 0.0f);

	return norm / normlen;
}

void Mesh::dice_subpatch(int subpatch_id, SubdParams& params)
{
	SubPatch& subpatch = subpatches[subpatch_id];
	Patch& patch = patches[subpatch.patch];

	params.shader = patches[subpatches[p].patch].shader;

	LinearQuadPatch quad_patch;
	LinearTrianglePatch tri_patch;
	ccl::Patch* subd_patch;

	//Attribute *attr_vF = attributes.find(ATTR_STD_FACE_NORMAL);
	//float3 *vF = attr_vF->data_float3();

	Attribute *attr_vN = attributes.find(ATTR_STD_VERTEX_NORMAL);
	float3 *vN = attr_vN->data_float3();

	if(patch.is_quad()) {
		for(int i = 0; i < 4; i++) {
			quad_patch.hull[i] = verts[patch.v[i]];
		}

		if(patch.smooth) {
			for(int i = 0; i < 4; i++)
				quad_patch.normals[i] = vN[patch.v[i]];
		}
		else {
			for(int i = 0; i < 4; i++)
				quad_patch.normals[i] = patch_normal(this, subpatch.patch);
		}

		swap(quad_patch.hull[2], quad_patch.hull[3]);
		swap(quad_patch.normals[2], quad_patch.normals[3]);

		subd_patch = &quad_patch;
	}
	else {
		for(int i = 0; i < 3; i++) {
			tri_patch.hull[i] = verts[patch.v[i]];
		}

		if(patch.smooth) {
			for(int i = 0; i < 3; i++)
				tri_patch.normals[i] = vN[patch.v[i]];
		}
		else {
			for(int i = 0; i < 3; i++)
				tri_patch.normals[i] = patch_normal(this, subpatch.patch);
		}

		subd_patch = &tri_patch;
	}

	if(subpatch.is_quad()) {
		QuadDice dice(params);

		QuadDice::SubPatch sub;
		QuadDice::EdgeFactors ef;

		sub.patch = subd_patch;
		sub.P00 = subpatch.uv[0];
		sub.P10 = subpatch.uv[1];
		sub.P01 = subpatch.uv[2];
		sub.P11 = subpatch.uv[3];

		ef.tu0 = subpatch.edge_factors[0];
		ef.tu1 = subpatch.edge_factors[1];
		ef.tv0 = subpatch.edge_factors[2];
		ef.tv1 = subpatch.edge_factors[3];

		dice.dice(sub, ef);
	}
	else {
		TriangleDice dice(params);

		TriangleDice::SubPatch sub;
		TriangleDice::EdgeFactors ef;

		sub.patch = subd_patch;
		sub.Pu = subpatch.uv[0];
		sub.Pv = subpatch.uv[1];
		sub.Pw = subpatch.uv[2];

		ef.tu = subpatch.edge_factors[0];
		ef.tv = subpatch.edge_factors[1];
		ef.tw = subpatch.edge_factors[2];

		dice.dice(sub, ef);
	}
}

void Mesh::tessellate(DiagSplit *split)
{
	//add_face_normals();

	//Attribute *attr_vF = attributes.find(ATTR_STD_FACE_NORMAL);
	//float3 *vF = attr_vF->data_float3();

	Attribute *attr_vN = attributes.find(ATTR_STD_VERTEX_NORMAL);
	float3 *vN = attr_vN->data_float3();

	for(int p = 0; p < patches.size(); p++) {
		if(patches[p].is_quad()) {
			LinearQuadPatch patch;

			for(int i = 0; i < 4; i++) {
				patch.hull[i] = verts[patches[p].v[i]];
			}

			if(patches[p].smooth) {
				for(int i = 0; i < 4; i++)
					patch.normals[i] = vN[patches[p].v[i]];
			}
			else {
				for(int i = 0; i < 4; i++)
					patch.normals[i] = patch_normal(this, p);
			}

			swap(patch.hull[2], patch.hull[3]);
			swap(patch.normals[2], patch.normals[3]);

			split->split_quad(&patch);
		}
		else {
			LinearTrianglePatch patch;

			for(int i = 0; i < 3; i++) {
				patch.hull[i] = verts[patches[p].v[i]];
			}

			if(patches[p].smooth) {
				for(int i = 0; i < 3; i++)
					patch.normals[i] = vN[patches[p].v[i]];
			}
			else {
				for(int i = 0; i < 3; i++)
					patch.normals[i] = patch_normal(this, p);
			}

			split->split_triangle(&patch);
		}

		for(int i = 0; i < split->subpatches_quad.size(); i++) {
			QuadDice::SubPatch& sub = split->subpatches_quad[i];
			QuadDice::EdgeFactors& ef = split->edgefactors_quad[i];

			SubPatch subpatch;
			subpatch.patch = p;

			subpatch.uv[0] = sub.P00;
			subpatch.uv[1] = sub.P10;
			subpatch.uv[2] = sub.P01;
			subpatch.uv[3] = sub.P11;

			subpatch.edge_factors[0] = max(ef.tu0, 1);
			subpatch.edge_factors[1] = max(ef.tu1, 1);
			subpatch.edge_factors[2] = max(ef.tv0, 1);
			subpatch.edge_factors[3] = max(ef.tv1, 1);

			subpatches.push_back(subpatch);
		}

		split->subpatches_quad.clear();
		split->edgefactors_quad.clear();

		for(int i = 0; i < split->subpatches_triangle.size(); i++) {
			TriangleDice::SubPatch& sub = split->subpatches_triangle[i];
			TriangleDice::EdgeFactors& ef = split->edgefactors_triangle[i];

			SubPatch subpatch;
			subpatch.patch = p;

			subpatch.uv[0] = sub.Pu;
			subpatch.uv[1] = sub.Pv;
			subpatch.uv[2] = sub.Pw;

			subpatch.edge_factors[0] = max(ef.tu, 1);
			subpatch.edge_factors[1] = max(ef.tv, 1);
			subpatch.edge_factors[2] = max(ef.tw, 1);
			subpatch.edge_factors[3] = -1;

			subpatches.push_back(subpatch);
		}

		split->subpatches_triangle.clear();
		split->edgefactors_triangle.clear();
	}

	for(int p = 0; p < subpatches.size(); p++) {
		dice_subpatch(p, split->params);
	}
}

CCL_NAMESPACE_END

