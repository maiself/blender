/*
 * Adapted from code copyright 2009-2010 NVIDIA Corporation
 * Modifications Copyright 2011, Blender Foundation.
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

#include "bvh/bvh.h"
#include "bvh/bvh_node.h"
#include "bvh/bvh_binning.h"

#include "util_bvh.h"
#include "util_math.h"

CCL_NAMESPACE_BEGIN

__forceinline void prefetch_L2 (const void* /*ptr*/) { }

#define PRIMITIVE_COST 1.0f
#define NODE_COST 1.0f
#define LEAF_MAX_TRIANGLES 8

struct BVHStackEntry
{
	const BVHNode *node;
	int idx;

	BVHStackEntry(const BVHNode* n = 0, int i = 0)
	: node(n), idx(i)
	{
	}

	int encodeIdx() const
	{
		return (node->is_leaf())? ~idx: idx;
	}
};

static BVHNode* util_bvh_build_node(const BVHObjectBinning& range, int level, BVHReference* references, uint4* triangles)
{
	size_t size = range.size();
	float leafSAH = PRIMITIVE_COST * range.leafSAH;
	float splitSAH = NODE_COST * range.bounds().half_area() + PRIMITIVE_COST * range.splitSAH;

	if(size == 0 || level > 0) {
		/* make leaf node when threshold reached or SAH tells us */
		if((size <= 1 || level >= 64) || (size <= LEAF_MAX_TRIANGLES && leafSAH < splitSAH))
			return new LeafNode(range.bounds(), 0, range.start(), range.start() + range.size());
	}

	/* perform split */
	BVHObjectBinning left, right;
	range.split(references, left, right);

	/* create inner node. */
	BVHNode *leftnode = util_bvh_build_node(left, level + 1, references, triangles);
	BVHNode *rightnode = util_bvh_build_node(right, level + 1, references, triangles);

	return new InnerNode(range.bounds(), leftnode, rightnode);
}

static void util_bvh_pack_leaf(float4* pack, const BVHStackEntry& e, const LeafNode *leaf)
{
	float4 data[BVH_NODE_LEAF_SIZE];
	memset(data, 0, sizeof(data));

	data[0].x = __int_as_float(leaf->m_lo);
	data[0].y = __int_as_float(leaf->m_hi);

	memcpy(&pack[e.idx], data, sizeof(float4)*BVH_NODE_LEAF_SIZE);
}

static void util_bvh_pack_inner(float4* pack, const BVHStackEntry& e, const BVHStackEntry& e0, const BVHStackEntry& e1)
{
	const BoundBox& b0 = e0.node->m_bounds;
	const BoundBox& b1 = e1.node->m_bounds;

	int4 data[BVH_NODE_SIZE] =
	{
		make_int4(__float_as_int(b0.min.x), __float_as_int(b1.min.x), __float_as_int(b0.max.x), __float_as_int(b1.max.x)),
		make_int4(__float_as_int(b0.min.y), __float_as_int(b1.min.y), __float_as_int(b0.max.y), __float_as_int(b1.max.y)),
		make_int4(__float_as_int(b0.min.z), __float_as_int(b1.min.z), __float_as_int(b0.max.z), __float_as_int(b1.max.z)),
		make_int4(e0.encodeIdx(), e1.encodeIdx(), 0, 0)
	};

	memcpy(&pack[e.idx], data, sizeof(int4)*BVH_NODE_SIZE);
}

size_t util_bvh_build(float4* bvh_data, float4* verts, uint4* triangles, uint num_triangles, uint size_allocated)
{
	BVHRange root;
	vector<BVHReference> references;

	/* add references */
	{
		/* reserve space for references */
		references.reserve(num_triangles);

		/* add references from triangles */
		BoundBox bounds = BoundBox::empty, center = BoundBox::empty;

		for(int i = 0; i < num_triangles; i++) {
			uint4 tri = triangles[i];
			BoundBox tbounds = BoundBox::empty;

			tbounds.grow(float4_to_float3(verts[tri.x]));
			tbounds.grow(float4_to_float3(verts[tri.y]));
			tbounds.grow(float4_to_float3(verts[tri.z]));

			if(tbounds.valid()) {
				references.push_back(BVHReference(tbounds, i, 0, 0));
				bounds.grow(tbounds);
				center.grow(tbounds.center2());
			}
		}

		/* happens mostly on empty meshes */
		if(!bounds.valid())
			bounds.grow(make_float3(0.0f, 0.0f, 0.0f));

		root = BVHRange(bounds, center, 0, references.size());
	}
	
	BVHNode *rootnode;

	/* build recursively */
	{
		BVHObjectBinning rootbin(root, (references.size())? &references[0]: NULL);

		rootnode = util_bvh_build_node(rootbin, 0, &references[0], triangles);
	}

	/* reorder triangles */
	{
		vector<uint4> tmp_triangles(num_triangles);
		memcpy(&tmp_triangles[0], triangles, sizeof(uint4)*num_triangles);

		for(int i = 0; i < references.size(); i++) {
			triangles[i] = tmp_triangles[references[i].prim_index()];
		}
	}

	/* pack nodes */
	size_t size;
	{
		size_t tot_node_size = rootnode->getSubtreeSize(BVH_STAT_NODE_COUNT);
		size_t leaf_node_size = rootnode->getSubtreeSize(BVH_STAT_LEAF_COUNT);
		size_t node_size = tot_node_size - leaf_node_size;

		size = (node_size+1)*4 + leaf_node_size;
		assert(size <= size_allocated);

		int nextNodeIdx = 1, nextLeafNodeIdx = (node_size+1)*4;

		vector<BVHStackEntry> stack;
		stack.reserve(64*2);

		if(rootnode->is_leaf())
			stack.push_back(BVHStackEntry(rootnode, nextLeafNodeIdx++));
		else
			stack.push_back(BVHStackEntry(rootnode, (nextNodeIdx++)*4));

		while(stack.size()) {
			BVHStackEntry e = stack.back();
			stack.pop_back();

			if(e.node->is_leaf()) {
				/* leaf node */
				const LeafNode* leaf = reinterpret_cast<const LeafNode*>(e.node);
				util_bvh_pack_leaf(bvh_data, e, leaf);
			}
			else {
				/* innner node */
				int idx0 = (e.node->get_child(0)->is_leaf())? (nextLeafNodeIdx++) : (nextNodeIdx++)*4;
				int idx1 = (e.node->get_child(1)->is_leaf())? (nextLeafNodeIdx++) : (nextNodeIdx++)*4;

				assert(idx0+(e.node->get_child(0)->is_leaf()? 1: 4) <= size);
				assert(idx1+(e.node->get_child(1)->is_leaf()? 1: 4) <= size);

				stack.push_back(BVHStackEntry(e.node->get_child(0), idx0));
				stack.push_back(BVHStackEntry(e.node->get_child(1), idx1));

				util_bvh_pack_inner(bvh_data, e, stack[stack.size()-2], stack[stack.size()-1]);
			}
		}

		/* root index to start traversal at, to handle case of single leaf node */
		bvh_data[0].x = __int_as_float(rootnode->is_leaf()? -5: 4);
	}

	rootnode->deleteSubtree();

	return size;
}

CCL_NAMESPACE_END


