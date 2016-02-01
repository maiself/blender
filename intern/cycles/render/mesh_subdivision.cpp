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


#include <opensubdiv/far/topologyDescriptor.h>
#include <opensubdiv/far/primvarRefiner.h>
#include <opensubdiv/far/patchTableFactory.h>
#include <opensubdiv/far/patchMap.h>
#include <opensubdiv/far/ptexIndices.h>

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

using namespace OpenSubdiv;

CCL_NAMESPACE_BEGIN

struct OsdVertex {
	float3 v;

	OsdVertex() {}

	void Clear(void* = 0) {
		v = make_float3(0.0f, 0.0f, 0.0f);
	}

	void AddWithWeight(OsdVertex const& src, float weight) {
		v += src.v * weight;
	}
};

struct MeshOsdData {
	Far::PatchTable* patch_table;
	Far::PatchMap* patch_map;
	Far::PtexIndices* ptex_indices;
	vector<OsdVertex> verts;

	vector<int> num_verts_per_face;
	vector<int> face_verts;
};

void Mesh::free_osd_data()
{
	if(!osd_data)
		return;

	delete osd_data->patch_table;
	delete osd_data->patch_map;
	delete osd_data->ptex_indices;

	delete osd_data;
	osd_data = NULL;
}

void Mesh::update_osd()
{
	assert(!osd_data);

	if(subdivision_type != Mesh::SUBDIVISION_CATMALL_CLARK)
		return;

	osd_data = new MeshOsdData;

	Sdc::SchemeType type = Sdc::SCHEME_CATMARK;

	Sdc::Options options;
	options.SetVtxBoundaryInterpolation(Sdc::Options::VTX_BOUNDARY_EDGE_ONLY);

	Far::TopologyDescriptor desc;
	desc.numVertices = verts.size();
	desc.numFaces = patches.size();

	size_t num_face_verts = 0;
	osd_data->num_verts_per_face.resize(patches.size());

	for(int i = 0; i < patches.size(); i++) {
		osd_data->num_verts_per_face[i] = patches[i].is_quad() ? 4 : 3;
		num_face_verts += osd_data->num_verts_per_face[i];
	}

	desc.numVertsPerFace = &osd_data->num_verts_per_face[0];

	osd_data->face_verts.resize(num_face_verts);

	int* fv = &osd_data->face_verts[0];
	for(int i = 0; i < patches.size(); i++) {
		Patch& patch = patches[i];

		if(patch.is_quad()) {
			*fv++ = patch.v[0];
			*fv++ = patch.v[1];
			*fv++ = patch.v[2];
			*fv++ = patch.v[3];
		}
		else {
			*fv++ = patch.v[0];
			*fv++ = patch.v[1];
			*fv++ = patch.v[2];
		}
	}

	desc.vertIndicesPerFace = &osd_data->face_verts[0];

	Far::TopologyRefiner* refiner = Far::TopologyRefinerFactory<Far::TopologyDescriptor>::Create(
		desc, Far::TopologyRefinerFactory<Far::TopologyDescriptor>::Options(type, options));

	int max_isolation = 10;
	refiner->RefineAdaptive(Far::TopologyRefiner::AdaptiveOptions(max_isolation));

	Far::PatchTableFactory::Options patch_options;
	patch_options.endCapType = Far::PatchTableFactory::Options::ENDCAP_GREGORY_BASIS;

	osd_data->patch_table = Far::PatchTableFactory::Create(*refiner, patch_options);

	int num_refiner_verts = refiner->GetNumVerticesTotal();
	int num_local_points = osd_data->patch_table->GetNumLocalPoints();

	osd_data->verts.resize(num_refiner_verts + num_local_points);
	for(int i = 0; i < verts.size(); i++) {
		osd_data->verts[i].v = verts[i];
	}

	OsdVertex* src = &osd_data->verts[0];
	for(int i = 0; i < refiner->GetMaxLevel(); i++) {
		OsdVertex* dest = src + refiner->GetLevel(i).GetNumVertices();
		Far::PrimvarRefiner(*refiner).Interpolate(i+1, src, dest);
		src = dest;
	}

	osd_data->patch_table->ComputeLocalPointValues(&osd_data->verts[0], &osd_data->verts[num_refiner_verts]);

	osd_data->patch_map = new Far::PatchMap(*osd_data->patch_table);
	osd_data->ptex_indices = new Far::PtexIndices(*refiner);
}

struct OsdPatch : Patch {
	Mesh* mesh;
	int patch;

	void eval(float3 *P, float3 *dPdu, float3 *dPdv, float3 *N, float u, float v)
	{
		int patch_ = patch;
		patch_uv_to_ptex_uv(patch_, u, v);

		const Far::PatchTable::PatchHandle* handle = mesh->osd_data->patch_map->FindPatch(patch_, u, v);
		assert(handle);

		float p_weights[20], du_weights[20], dv_weights[20];
		mesh->osd_data->patch_table->EvaluateBasis(*handle, u, v, p_weights, du_weights, dv_weights);

		Far::ConstIndexArray cv = mesh->osd_data->patch_table->GetPatchVertices(*handle);

		float3 du, dv;
		if(P) *P = make_float3(0.0f, 0.0f, 0.0f);
		du = make_float3(0.0f, 0.0f, 0.0f);
		dv = make_float3(0.0f, 0.0f, 0.0f);

		for(int i = 0; i < cv.size(); i++) {
			float3 p = mesh->osd_data->verts[cv[i]].v;

			if(P) *P += p * p_weights[i];
			du += p * du_weights[i];
			dv += p * dv_weights[i];
		}

		if(dPdu) *dPdu = du;
		if(dPdv) *dPdv = dv;
		if(N) *N = normalize(cross(du, dv));
	}

	void patch_uv_to_ptex_uv(int& patch, float& u, float& v)
	{
		bool is_quad = mesh->patches[patch].is_quad();
		patch = mesh->osd_data->ptex_indices->GetFaceId(patch);

		if(!is_quad) {
			float w = 1.0f - u - v;

			int quad = util_max_axis(make_float3(u, v, w));
			patch += quad;

			float u_, v_;

			switch(quad) {
				case 0: { u_ = v; v_ = w; break; }
				case 1: { u_ = w; v_ = u; break; }
				case 2: { u_ = u; v_ = v; break; }
				default: assert(0);
			}

			u_ *= 2.0f;
			v_ *= 2.0f;

			float s = sqrtf(u_*u_ + v_*v_ - 2.0f*u_*v_ - 6.0f*u_ - 6.0f*v_ + 9.0f);

			u = u_ - v_ + s - 3.0f;
			if(fabsf(u) > 0.000001f)
				u = clamp(3.0f*(u_ + v_ + s - 3.0f)/u, 0.0f, 1.0f);
			else
				u = u_;

			v = v_ - u_ + s - 3.0f;
			if(fabsf(v) > 0.000001f)
				v = clamp(3.0f*(v_ + u_ + s - 3.0f)/v, 0.0f, 1.0f);
			else
				v = v_;
		}
	}

	bool is_triangle() { return !mesh->patches[patch].is_quad(); }
	BoundBox bound() { return BoundBox::empty; }
};

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

	OsdPatch osd_patch;
	LinearQuadPatch quad_patch;
	LinearTrianglePatch tri_patch;
	ccl::Patch* subd_patch;

	if(subdivision_type == SUBDIVISION_CATMALL_CLARK) {
		osd_patch.mesh = this;
		osd_patch.patch = subpatches[subpatch_id].patch;

		subd_patch = &osd_patch;
	}
	else {
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
	update_osd();

	Attribute *attr_vN = attributes.find(ATTR_STD_VERTEX_NORMAL);
	float3 *vN = attr_vN->data_float3();

	for(int p = 0; p < patches.size(); p++) {
		if(subdivision_type == SUBDIVISION_CATMALL_CLARK) {
			OsdPatch patch;
			patch.mesh = this;
			patch.patch = p;

			if(patches[p].is_quad()) 
				split->split_quad(&patch);
			else 
				split->split_triangle(&patch);
		}
		else {
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

