/*
 * Copyright 2011-2015 Blender Foundation
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

#include "split/kernel_data_init.h"

__kernel void kernel_ocl_path_trace_data_init(
        ccl_global char *globals,
        ccl_global char *sd_DL_shadow,
        ccl_constant KernelData *data,
        ccl_global float *per_sample_output_buffers,
        ccl_global uint *rng_state,
        ccl_global uint *rng_coop,                   /* rng array to store rng values for all rays */
        ccl_global float3 *throughput_coop,          /* throughput array to store throughput values for all rays */
        ccl_global float *L_transparent_coop,        /* L_transparent array to store L_transparent values for all rays */
        PathRadiance *PathRadiance_coop,             /* PathRadiance array to store PathRadiance values for all rays */
        ccl_global Ray *Ray_coop,                    /* Ray array to store Ray information for all rays */
        ccl_global PathState *PathState_coop,        /* PathState array to store PathState information for all rays */
        ccl_global Intersection *Intersection_coop_shadow,
        ccl_global PathState *state_shadow,
        ccl_global SubsurfaceIntersection *ss_isect,
        ccl_global SubsurfaceIndirectRays *SSS_coop,
        ccl_global char *ray_state,                  /* Stores information on current state of a ray */

#define KERNEL_TEX(type, ttype, name) \
        ccl_global type *name,
#include "../../kernel_textures.h"

        int start_sample, int sx, int sy, int sw, int sh, int offset, int stride,
        int rng_state_offset_x,
        int rng_state_offset_y,
        int rng_state_stride,
        ccl_global int *Queue_data,                  /* Memory for queues */
        ccl_global int *Queue_index,                 /* Tracks the number of elements in queues */
        int queuesize,                               /* size (capacity) of the queue */
        ccl_global char *use_queues_flag,            /* flag to decide if scene-intersect kernel should use queues to fetch ray index */
        ccl_global unsigned int *work_array,         /* work array to store which work each ray belongs to */
#ifdef __WORK_STEALING__
        ccl_global unsigned int *work_pool_wgs,      /* Work pool for each work group */
        unsigned int num_samples,                    /* Total number of samples per pixel */
#endif
#ifdef __KERNEL_DEBUG__
        DebugData *debugdata_coop,
#endif
        int parallel_samples)                        /* Number of samples to be processed in parallel */
{
	kernel_data_init((KernelGlobals *)globals,
	                 (ShaderData *)sd_DL_shadow,
	                 data,
	                 per_sample_output_buffers,
	                 rng_state,
	                 rng_coop,
	                 throughput_coop,
	                 L_transparent_coop,
	                 PathRadiance_coop,
	                 Ray_coop,
	                 PathState_coop,
	                 Intersection_coop_shadow,
	                 state_shadow,
	                 ss_isect,
	                 SSS_coop,
	                 ray_state,

#define KERNEL_TEX(type, ttype, name) name,
#include "../../kernel_textures.h"

	                 start_sample, sx, sy, sw, sh, offset, stride,
	                 rng_state_offset_x,
	                 rng_state_offset_y,
	                 rng_state_stride,
	                 Queue_data,
	                 Queue_index,
	                 queuesize,
	                 use_queues_flag,
	                 work_array,
#ifdef __WORK_STEALING__
	                 work_pool_wgs,
	                 num_samples,
#endif
#ifdef __KERNEL_DEBUG__
	                 debugdata_coop,
#endif
	                 parallel_samples);
}
/*
 * Copyright 2011-2015 Blender Foundation
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

#include "split/kernel_scene_intersect.h"

__kernel void kernel_ocl_path_trace_scene_intersect(
        ccl_global char *kg,
        ccl_constant KernelData *data,
        ccl_global uint *rng_coop,
        ccl_global Ray *Ray_coop,              /* Required for scene_intersect */
        ccl_global PathState *PathState_coop,  /* Required for scene_intersect */
        ccl_global Intersection *Intersection_coop,       /* Required for scene_intersect */
        ccl_global char *ray_state,            /* Denotes the state of each ray */
        int sw, int sh,
        ccl_global int *Queue_data,            /* Memory for queues */
        ccl_global int *Queue_index,           /* Tracks the number of elements in queues */
        int queuesize,                         /* Size (capacity) of queues */
        ccl_global char *use_queues_flag,      /* used to decide if this kernel should use
                                                * queues to fetch ray index */
#ifdef __KERNEL_DEBUG__
        DebugData *debugdata_coop,
#endif
        int parallel_samples)                  /* Number of samples to be processed in parallel */
{
	int x = get_global_id(0);
	int y = get_global_id(1);

	/* Fetch use_queues_flag */
	ccl_local char local_use_queues_flag;
	if(get_local_id(0) == 0 && get_local_id(1) == 0) {
		local_use_queues_flag = use_queues_flag[0];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	int ray_index;
	if(local_use_queues_flag) {
		int thread_index = get_global_id(1) * get_global_size(0) + get_global_id(0);
		ray_index = get_ray_index(thread_index,
		                          QUEUE_ACTIVE_AND_REGENERATED_RAYS,
		                          Queue_data,
		                          queuesize,
		                          0);

		if(ray_index == QUEUE_EMPTY_SLOT) {
			return;
		}
	} else {
		if(x < (sw * parallel_samples) && y < sh) {
			ray_index = x + y * (sw * parallel_samples);
		} else {
			return;
		}
	}

	kernel_scene_intersect((KernelGlobals *)kg,
	                       rng_coop,
	                       Ray_coop,
	                       PathState_coop,
	                       Intersection_coop,
	                       ray_state,
	                       sw, sh,
	                       use_queues_flag,
#ifdef __KERNEL_DEBUG__
	                       debugdata_coop,
#endif
	                       ray_index);
}
/*
 * Copyright 2011-2015 Blender Foundation
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

#include "split/kernel_lamp_emission.h"

__kernel void kernel_ocl_path_trace_lamp_emission(
        ccl_global char *kg,
        ccl_constant KernelData *data,
        ccl_global float3 *throughput_coop,    /* Required for lamp emission */
        PathRadiance *PathRadiance_coop,       /* Required for lamp emission */
        ccl_global Ray *Ray_coop,              /* Required for lamp emission */
        ccl_global PathState *PathState_coop,  /* Required for lamp emission */
        ccl_global Intersection *Intersection_coop,       /* Required for lamp emission */
        ccl_global char *ray_state,            /* Denotes the state of each ray */
        int sw, int sh,
        ccl_global int *Queue_data,            /* Memory for queues */
        ccl_global int *Queue_index,           /* Tracks the number of elements in queues */
        int queuesize,                         /* Size (capacity) of queues */
        ccl_global char *use_queues_flag,      /* Used to decide if this kernel should use
                                                * queues to fetch ray index
                                                */
        int parallel_samples)                  /* Number of samples to be processed in parallel */
{
	int x = get_global_id(0);
	int y = get_global_id(1);

	/* Fetch use_queues_flag. */
	ccl_local char local_use_queues_flag;
	if(get_local_id(0) == 0 && get_local_id(1) == 0) {
		local_use_queues_flag = use_queues_flag[0];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	int ray_index;
	if(local_use_queues_flag) {
		int thread_index = get_global_id(1) * get_global_size(0) + get_global_id(0);
		ray_index = get_ray_index(thread_index,
		                          QUEUE_ACTIVE_AND_REGENERATED_RAYS,
		                          Queue_data,
		                          queuesize,
		                          0);
		if(ray_index == QUEUE_EMPTY_SLOT) {
			return;
		}
	} else {
		if(x < (sw * parallel_samples) && y < sh) {
			ray_index = x + y * (sw * parallel_samples);
		} else {
			return;
		}
	}

	kernel_lamp_emission((KernelGlobals *)kg,
	                     throughput_coop,
	                     PathRadiance_coop,
	                     Ray_coop,
	                     PathState_coop,
	                     Intersection_coop,
	                     ray_state,
	                     sw, sh,
	                     use_queues_flag,
	                     ray_index);
}
/*
 * Copyright 2011-2015 Blender Foundation
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

#include "split/kernel_do_volume.h"

__kernel void kernel_ocl_path_trace_do_volume(
        ccl_global char *kg,
        ccl_constant KernelData *data,
        ccl_global char *sd,                   /* Output ShaderData structure to be filled */
        ccl_global float3 *throughput_coop,    /* Required for volume */
        PathRadiance *PathRadiance_coop,       /* Required for volume */
        ccl_global Ray *Ray_coop,              /* Required for volume */
        ccl_global PathState *PathState_coop,  /* Required for volume */
        ccl_global Intersection *Intersection_coop,       /* Required for volume */
        ccl_global uint *rng_coop,             /* Required for rbsdf calculation */
        ccl_global char *ray_state,            /* Denotes the state of each ray */
        int sw, int sh,
        ccl_global int *Queue_data,            /* Memory for queues */
        ccl_global int *Queue_index,           /* Tracks the number of elements in queues */
        int queuesize,                         /* Size (capacity) of queues */
        ccl_global char *use_queues_flag,      /* Used to decide if this kernel should use
                                                * queues to fetch ray index
                                                */
        int parallel_samples)                  /* Number of samples to be processed in parallel */
{
	int x = get_global_id(0);
	int y = get_global_id(1);

	/* We will empty this queue in this kernel. */
	if(get_global_id(0) == 0 && get_global_id(1) == 0) {
		Queue_index[QUEUE_ACTIVE_AND_REGENERATED_RAYS] = 0;
	}
	/* Fetch use_queues_flag. */
	ccl_local char local_use_queues_flag;
	if(get_local_id(0) == 0 && get_local_id(1) == 0) {
		local_use_queues_flag = use_queues_flag[0];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	int ray_index;
	if(local_use_queues_flag) {
		int thread_index = get_global_id(1) * get_global_size(0) + get_global_id(0);
		ray_index = get_ray_index(thread_index,
		                          QUEUE_ACTIVE_AND_REGENERATED_RAYS,
		                          Queue_data,
		                          queuesize,
		                          1);
		if(ray_index == QUEUE_EMPTY_SLOT) {
			return;
		}
	} else {
		if(x < (sw * parallel_samples) && y < sh) {
			ray_index = x + y * (sw * parallel_samples);
		} else {
			return;
		}
	}

#ifdef __VOLUME__
	kernel_do_volume((KernelGlobals *)kg,
			 SD_REF((ShaderData *)sd, ray_index),
			 throughput_coop,
			 PathRadiance_coop,
			 Ray_coop,
			 PathState_coop,
			 Intersection_coop,
			 rng_coop,
			 ray_state,
			 sw, sh,
			 use_queues_flag,
			 ray_index);
#endif

}
/*
 * Copyright 2011-2015 Blender Foundation
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

//#include "../../kernel_compat_opencl.h"
//#include "../../kernel_math.h"
//#include "../../kernel_types.h"
//#include "../../kernel_globals.h"
//#include "../../kernel_queues.h"

/*
 * The kernel "kernel_queue_enqueue" enqueues rays of
 * different ray state into their appropriate Queues;
 * 1. Rays that have been determined to hit the background from the
 * "kernel_scene_intersect" kernel
 * are enqueued in QUEUE_HITBG_BUFF_UPDATE_TOREGEN_RAYS;
 * 2. Rays that have been determined to be actively participating in path-iteration will be enqueued into QUEUE_ACTIVE_AND_REGENERATED_RAYS.
 *
 * The input and output of the kernel is as follows,
 *
 * ray_state -------------------------------------------|--- kernel_queue_enqueue --|--- Queue_data (QUEUE_ACTIVE_AND_REGENERATED_RAYS & QUEUE_HITBF_BUFF_UPDATE_TOREGEN_RAYS)
 * Queue_index(QUEUE_ACTIVE_AND_REGENERATED_RAYS) ------|                           |--- Queue_index (QUEUE_ACTIVE_AND_REGENERATED_RAYS & QUEUE_HITBF_BUFF_UPDATE_TOREGEN_RAYS)
 * Queue_index(QUEUE_HITBG_BUFF_UPDATE_TOREGEN_RAYS) ---|                           |
 * queuesize -------------------------------------------|                           |
 *
 * Note on Queues :
 * State of queues during the first time this kernel is called :
 * At entry,
 * Both QUEUE_ACTIVE_AND_REGENERATED_RAYS and QUEUE_HITBG_BUFF_UPDATE_TOREGEN_RAYS will be empty.
 * At exit,
 * QUEUE_ACTIVE_AND_REGENERATED_RAYS will be filled with RAY_ACTIVE rays
 * QUEUE_HITBF_BUFF_UPDATE_TOREGEN_RAYS will be filled with RAY_HIT_BACKGROUND rays.
 *
 * State of queue during other times this kernel is called :
 * At entry,
 * QUEUE_ACTIVE_AND_REGENERATED_RAYS will be empty.
 * QUEUE_HITBG_BUFF_UPDATE_TOREGEN_RAYS will contain RAY_TO_REGENERATE and RAY_UPDATE_BUFFER rays.
 * At exit,
 * QUEUE_ACTIVE_AND_REGENERATED_RAYS will be filled with RAY_ACTIVE rays.
 * QUEUE_HITBG_BUFF_UPDATE_TOREGEN_RAYS will be filled with RAY_TO_REGENERATE, RAY_UPDATE_BUFFER, RAY_HIT_BACKGROUND rays.
 */
__kernel void kernel_ocl_path_trace_queue_enqueue(
        ccl_global int *Queue_data,   /* Queue memory */
        ccl_global int *Queue_index,  /* Tracks the number of elements in each queue */
        ccl_global char *ray_state,   /* Denotes the state of each ray */
        int queuesize)                /* Size (capacity) of each queue */
{
	/* We have only 2 cases (Hit/Not-Hit) */
	ccl_local unsigned int local_queue_atomics[2];

	int lidx = get_local_id(1) * get_local_size(0) + get_local_id(0);
	int ray_index = get_global_id(1) * get_global_size(0) + get_global_id(0);

	if(lidx < 2 ) {
		local_queue_atomics[lidx] = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	int queue_number = -1;

	if(IS_STATE(ray_state, ray_index, RAY_HIT_BACKGROUND) ||
	   IS_STATE(ray_state, ray_index, RAY_UPDATE_BUFFER)) {
		queue_number = QUEUE_HITBG_BUFF_UPDATE_TOREGEN_RAYS;
	}
	else if(IS_STATE(ray_state, ray_index, RAY_ACTIVE) ||
		IS_STATE(ray_state, ray_index, RAY_REGENERATED)) {
		queue_number = QUEUE_ACTIVE_AND_REGENERATED_RAYS;
	}

	unsigned int my_lqidx;
	if(queue_number != -1) {
		my_lqidx = get_local_queue_index(queue_number, local_queue_atomics);
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if(lidx == 0) {
		local_queue_atomics[QUEUE_ACTIVE_AND_REGENERATED_RAYS] =
		        get_global_per_queue_offset(QUEUE_ACTIVE_AND_REGENERATED_RAYS,
		                                    local_queue_atomics,
		                                    Queue_index);
		local_queue_atomics[QUEUE_HITBG_BUFF_UPDATE_TOREGEN_RAYS] =
		        get_global_per_queue_offset(QUEUE_HITBG_BUFF_UPDATE_TOREGEN_RAYS,
		                                    local_queue_atomics,
		                                    Queue_index);
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	unsigned int my_gqidx;
	if(queue_number != -1) {
		my_gqidx = get_global_queue_index(queue_number,
		                                  queuesize,
		                                  my_lqidx,
		                                  local_queue_atomics);
		Queue_data[my_gqidx] = ray_index;
	}
}
/*
 * Copyright 2011-2015 Blender Foundation
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

#include "split/kernel_indirect_background.h"

__kernel void kernel_ocl_path_trace_indirect_background(
        ccl_global char *kg,
        ccl_constant KernelData *data,
        ccl_global float3 *throughput_coop,    /* Required for background hit processing */
        PathRadiance *PathRadiance_coop,       /* Required for background hit processing and buffer Update */
        ccl_global Ray *Ray_coop,              /* Required for background hit processing */
        ccl_global PathState *PathState_coop,  /* Required for background hit processing */
        ccl_global float *L_transparent_coop,  /* Required for background hit processing and buffer Update */
        ccl_global char *ray_state,            /* Stores information on the current state of a ray */
        ccl_global int *Queue_data,            /* Queues memory */
        ccl_global int *Queue_index,           /* Tracks the number of elements in each queue */
        int queuesize)                         /* Size (capacity) of each queue */
{
	ccl_local unsigned int local_queue_atomics;
	if(get_local_id(0) == 0 && get_local_id(1) == 0) {
		local_queue_atomics = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	int ray_index = get_global_id(1) * get_global_size(0) + get_global_id(0);
	ray_index = get_ray_index(ray_index,
	                          QUEUE_HITBG_BUFF_UPDATE_TOREGEN_RAYS,
	                          Queue_data,
	                          queuesize,
	                          0);

	kernel_indirect_background((KernelGlobals *)kg,
	                           throughput_coop,
	                           PathRadiance_coop,
	                           Ray_coop,
	                           PathState_coop,
	                           L_transparent_coop,
	                           ray_state,
	                           ray_index);
}
/*
 * Copyright 2011-2015 Blender Foundation
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

#include "split/kernel_shader_eval.h"

__kernel void kernel_ocl_path_trace_shader_eval(
        ccl_global char *kg,
        ccl_constant KernelData *data,
        ccl_global char *sd,                   /* Output ShaderData structure to be filled */
        ccl_global uint *rng_coop,             /* Required for rbsdf calculation */
        ccl_global Ray *Ray_coop,              /* Required for setting up shader from ray */
        ccl_global PathState *PathState_coop,  /* Required for all functions in this kernel */
        ccl_global Intersection *Intersection_coop,       /* Required for setting up shader from ray */
        ccl_global char *ray_state,            /* Denotes the state of each ray */
        ccl_global int *Queue_data,            /* queue memory */
        ccl_global int *Queue_index,           /* Tracks the number of elements in each queue */
        int queuesize)                         /* Size (capacity) of each queue */
{
	/* Enqeueue RAY_TO_REGENERATE rays into QUEUE_HITBG_BUFF_UPDATE_TOREGEN_RAYS queue. */
	ccl_local unsigned int local_queue_atomics;
	if(get_local_id(0) == 0 && get_local_id(1) == 0) {
		local_queue_atomics = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	int ray_index = get_global_id(1) * get_global_size(0) + get_global_id(0);
	ray_index = get_ray_index(ray_index,
	                          QUEUE_ACTIVE_AND_REGENERATED_RAYS,
	                          Queue_data,
	                          queuesize,
	                          0);

	if(ray_index == QUEUE_EMPTY_SLOT) {
		return;
	}

	char enqueue_flag = (IS_STATE(ray_state, ray_index, RAY_TO_REGENERATE)) ? 1 : 0;
	enqueue_ray_index_local(ray_index,
	                        QUEUE_HITBG_BUFF_UPDATE_TOREGEN_RAYS,
	                        enqueue_flag,
	                        queuesize,
	                        &local_queue_atomics,
	                        Queue_data,
	                        Queue_index);

	/* Continue on with shader evaluation. */
	kernel_shader_eval((KernelGlobals *)kg,
	                   SD_REF((ShaderData *)sd, ray_index),
	                   rng_coop,
	                   Ray_coop,
	                   PathState_coop,
	                   Intersection_coop,
	                   ray_state,
	                   ray_index);
}
/*
 * Copyright 2011-2015 Blender Foundation
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

#include "split/kernel_holdout_emission_blurring_pathtermination_ao.h"

__kernel void kernel_ocl_path_trace_holdout_emission_blurring_pathtermination_ao(
        ccl_global char *kg,
        ccl_constant KernelData *data,
        ccl_global char *sd,                   /* Required throughout the kernel except probabilistic path termination and AO */
        ccl_global float *per_sample_output_buffers,
        ccl_global uint *rng_coop,             /* Required for "kernel_write_data_passes" and AO */
        ccl_global float3 *throughput_coop,    /* Required for handling holdout material and AO */
        ccl_global float *L_transparent_coop,  /* Required for handling holdout material */
        PathRadiance *PathRadiance_coop,       /* Required for "kernel_write_data_passes" and indirect primitive emission */
        ccl_global PathState *PathState_coop,  /* Required throughout the kernel and AO */
        ccl_global Intersection *Intersection_coop,       /* Required for indirect primitive emission */
        ccl_global float3 *AOAlpha_coop,       /* Required for AO */
        ccl_global float3 *AOBSDF_coop,        /* Required for AO */
        ccl_global Ray *AOLightRay_coop,       /* Required for AO */
        int sw, int sh, int sx, int sy, int stride,
        ccl_global char *ray_state,            /* Denotes the state of each ray */
        ccl_global unsigned int *work_array,   /* Denotes the work that each ray belongs to */
        ccl_global int *Queue_data,            /* Queue memory */
        ccl_global int *Queue_index,           /* Tracks the number of elements in each queue */
        int queuesize,                         /* Size (capacity) of each queue */
#ifdef __WORK_STEALING__
        unsigned int start_sample,
#endif
        int parallel_samples)                  /* Number of samples to be processed in parallel */
{
	ccl_local unsigned int local_queue_atomics_bg;
	ccl_local unsigned int local_queue_atomics_ao;
	if(get_local_id(0) == 0 && get_local_id(1) == 0) {
		local_queue_atomics_bg = 0;
		local_queue_atomics_ao = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	char enqueue_flag = 0;
	char enqueue_flag_AO_SHADOW_RAY_CAST = 0;
	int ray_index = get_global_id(1) * get_global_size(0) + get_global_id(0);
	ray_index = get_ray_index(ray_index,
	                          QUEUE_ACTIVE_AND_REGENERATED_RAYS,
	                          Queue_data,
	                          queuesize,
	                          0);

#ifdef __COMPUTE_DEVICE_GPU__
	/* If we are executing on a GPU device, we exit all threads that are not
	 * required.
	 *
	 * If we are executing on a CPU device, then we need to keep all threads
	 * active since we have barrier() calls later in the kernel. CPU devices,
	 * expect all threads to execute barrier statement.
	 */
	if(ray_index == QUEUE_EMPTY_SLOT) {
		return;
	}
#endif  /* __COMPUTE_DEVICE_GPU__ */

#ifndef __COMPUTE_DEVICE_GPU__
	if(ray_index != QUEUE_EMPTY_SLOT) {
#endif
		kernel_holdout_emission_blurring_pathtermination_ao(
		        (KernelGlobals *)kg,
		        SD_REF((ShaderData *)sd, ray_index),
		        per_sample_output_buffers,
		        rng_coop,
		        throughput_coop,
		        L_transparent_coop,
		        PathRadiance_coop,
		        PathState_coop,
		        Intersection_coop,
		        AOAlpha_coop,
		        AOBSDF_coop,
		        AOLightRay_coop,
		        sw, sh, sx, sy, stride,
		        ray_state,
		        work_array,
#ifdef __WORK_STEALING__
		        start_sample,
#endif
		        parallel_samples,
		        ray_index,
		        &enqueue_flag,
		        &enqueue_flag_AO_SHADOW_RAY_CAST);
#ifndef __COMPUTE_DEVICE_GPU__
	}
#endif

	/* Enqueue RAY_UPDATE_BUFFER rays. */
	enqueue_ray_index_local(ray_index,
	                        QUEUE_HITBG_BUFF_UPDATE_TOREGEN_RAYS,
	                        enqueue_flag,
	                        queuesize,
	                        &local_queue_atomics_bg,
	                        Queue_data,
	                        Queue_index);

#ifdef __AO__
	/* Enqueue to-shadow-ray-cast rays. */
	enqueue_ray_index_local(ray_index,
	                        QUEUE_SHADOW_RAY_CAST_AO_RAYS,
	                        enqueue_flag_AO_SHADOW_RAY_CAST,
	                        queuesize,
	                        &local_queue_atomics_ao,
	                        Queue_data,
	                        Queue_index);
#endif
}
/*
 * Copyright 2011-2015 Blender Foundation
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

#include "split/kernel_subsurface_scatter.h"

__kernel void kernel_ocl_path_trace_subsurface_scatter(
        ccl_global char *kg,
        ccl_constant KernelData *data,
        ccl_global char *sd,                          /* Required for direct lighting */
        ccl_global uint *rng_coop,                    /* Required for direct lighting */
        ccl_global PathState *PathState_coop,         /* Required for direct lighting */
        ccl_global Ray *Ray_coop,                     /* Required for direct lighting */
        ccl_global float3 *throughput_coop,           /* Required for direct lighting */
        PathRadiance *PathRadiance_coop,              /* Required for direct lighting */
        ccl_global SubsurfaceIndirectRays *SSS_coop,  /* Required for direct lighting */
        ccl_global char *ray_state,                   /* Denotes the state of each ray */
        ccl_global int *Queue_data,                   /* Queue memory */
        ccl_global int *Queue_index,                  /* Tracks the number of elements in each queue */
        int queuesize)                                /* Size (capacity) of each queue */
{
#ifdef __SUBSURFACE__
	ccl_local unsigned int local_queue_atomics;
	if(get_local_id(0) == 0 && get_local_id(1) == 0) {
		local_queue_atomics = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	char enqueue_flag = 0;
	int ray_index = get_global_id(1) * get_global_size(0) + get_global_id(0);
	ray_index = get_ray_index(ray_index,
	                          QUEUE_ACTIVE_AND_REGENERATED_RAYS,
	                          Queue_data,
	                          queuesize,
	                          0);

#ifdef __COMPUTE_DEVICE_GPU__
	if(ray_index == QUEUE_EMPTY_SLOT) {
		return;
	}
#endif

#ifndef __COMPUTE_DEVICE_GPU__
	if(ray_index != QUEUE_EMPTY_SLOT) {
#endif
		enqueue_flag = kernel_subsurface_scatter((KernelGlobals *)kg,
		                                         SD_REF((ShaderData *)sd, ray_index),
		                                         rng_coop,
		                                         PathState_coop,
		                                         Ray_coop,
		                                         throughput_coop,
		                                         PathRadiance_coop,
		                                         SSS_coop,
		                                         ray_state,
		                                         ray_index);
#ifndef __COMPUTE_DEVICE_GPU__
	}
#endif

	/* Enqueue RAY_UPDATE_BUFFER rays. */
	enqueue_ray_index_local(ray_index,
	                        QUEUE_HITBG_BUFF_UPDATE_TOREGEN_RAYS,
	                        enqueue_flag,
	                        queuesize,
	                        &local_queue_atomics,
	                        Queue_data,
	                        Queue_index);

#endif /* __SUBSURFACE__ */
}
/*
 * Copyright 2011-2015 Blender Foundation
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

#include "split/kernel_direct_lighting.h"

__kernel void kernel_ocl_path_trace_direct_lighting(
        ccl_global char *kg,
        ccl_constant KernelData *data,
        ccl_global char *sd,                    /* Required for direct lighting */
        ccl_global uint *rng_coop,              /* Required for direct lighting */
        ccl_global PathState *PathState_coop,   /* Required for direct lighting */
        ccl_global int *ISLamp_coop,            /* Required for direct lighting */
        ccl_global Ray *LightRay_coop,          /* Required for direct lighting */
        ccl_global BsdfEval *BSDFEval_coop,     /* Required for direct lighting */
        ccl_global char *ray_state,             /* Denotes the state of each ray */
        ccl_global int *Queue_data,             /* Queue memory */
        ccl_global int *Queue_index,            /* Tracks the number of elements in each queue */
        int queuesize)                          /* Size (capacity) of each queue */
{
	ccl_local unsigned int local_queue_atomics;
	if(get_local_id(0) == 0 && get_local_id(1) == 0) {
		local_queue_atomics = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	char enqueue_flag = 0;
	int ray_index = get_global_id(1) * get_global_size(0) + get_global_id(0);
	ray_index = get_ray_index(ray_index,
	                          QUEUE_ACTIVE_AND_REGENERATED_RAYS,
	                          Queue_data,
	                          queuesize,
	                          0);

#ifdef __COMPUTE_DEVICE_GPU__
	/* If we are executing on a GPU device, we exit all threads that are not
	 * required.
	 *
	 * If we are executing on a CPU device, then we need to keep all threads
	 * active since we have barrier() calls later in the kernel. CPU devices,
	 * expect all threads to execute barrier statement.
	 */
	if(ray_index == QUEUE_EMPTY_SLOT) {
		return;
	}
#endif

#ifndef __COMPUTE_DEVICE_GPU__
	if(ray_index != QUEUE_EMPTY_SLOT) {
#endif
		enqueue_flag = kernel_direct_lighting((KernelGlobals *)kg,
		                                      SD_REF((ShaderData *)sd, ray_index),
		                                      rng_coop,
		                                      PathState_coop,
		                                      ISLamp_coop,
		                                      LightRay_coop,
		                                      BSDFEval_coop,
		                                      ray_state,
		                                      ray_index);

#ifndef __COMPUTE_DEVICE_GPU__
	}
#endif

#ifdef __EMISSION__
	/* Enqueue RAY_SHADOW_RAY_CAST_DL rays. */
	enqueue_ray_index_local(ray_index,
	                        QUEUE_SHADOW_RAY_CAST_DL_RAYS,
	                        enqueue_flag,
	                        queuesize,
	                        &local_queue_atomics,
	                        Queue_data,
	                        Queue_index);
#endif
}
/*
 * Copyright 2011-2015 Blender Foundation
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

#include "split/kernel_shadow_blocked.h"

__kernel void kernel_ocl_path_trace_shadow_blocked(
        ccl_global char *kg,
        ccl_constant KernelData *data,
        ccl_global PathState *PathState_coop,  /* Required for shadow blocked */
        ccl_global Ray *LightRay_dl_coop,      /* Required for direct lighting's shadow blocked */
        ccl_global Ray *LightRay_ao_coop,      /* Required for AO's shadow blocked */
        ccl_global char *ray_state,
        ccl_global int *Queue_data,            /* Queue memory */
        ccl_global int *Queue_index,           /* Tracks the number of elements in each queue */
        int queuesize)                         /* Size (capacity) of each queue */
{
	int lidx = get_local_id(1) * get_local_id(0) + get_local_id(0);

	ccl_local unsigned int ao_queue_length;
	ccl_local unsigned int dl_queue_length;
	if(lidx == 0) {
		ao_queue_length = Queue_index[QUEUE_SHADOW_RAY_CAST_AO_RAYS];
		dl_queue_length = Queue_index[QUEUE_SHADOW_RAY_CAST_DL_RAYS];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	/* flag determining if the current ray is to process shadow ray for AO or DL */
	char shadow_blocked_type = -1;

	int ray_index = QUEUE_EMPTY_SLOT;
	int thread_index = get_global_id(1) * get_global_size(0) + get_global_id(0);
	if(thread_index < ao_queue_length + dl_queue_length) {
		if(thread_index < ao_queue_length) {
			ray_index = get_ray_index(thread_index, QUEUE_SHADOW_RAY_CAST_AO_RAYS, Queue_data, queuesize, 1);
			shadow_blocked_type = RAY_SHADOW_RAY_CAST_AO;
		} else {
			ray_index = get_ray_index(thread_index - ao_queue_length, QUEUE_SHADOW_RAY_CAST_DL_RAYS, Queue_data, queuesize, 1);
			shadow_blocked_type = RAY_SHADOW_RAY_CAST_DL;
		}
	}

	if(ray_index == QUEUE_EMPTY_SLOT)
		return;

	kernel_shadow_blocked((KernelGlobals *)kg,
	                      PathState_coop,
	                      LightRay_dl_coop,
	                      LightRay_ao_coop,
	                      ray_state,
	                      shadow_blocked_type,
	                      ray_index);
}
/*
 * Copyright 2011-2015 Blender Foundation
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

#include "split/kernel_next_iteration_setup.h"

__kernel void kernel_ocl_path_trace_next_iteration_setup(
        ccl_global char *kg,
        ccl_constant KernelData *data,
        ccl_global char *sd,                  /* Required for setting up ray for next iteration */
        ccl_global uint *rng_coop,            /* Required for setting up ray for next iteration */
        ccl_global float3 *throughput_coop,   /* Required for setting up ray for next iteration */
        PathRadiance *PathRadiance_coop,      /* Required for setting up ray for next iteration */
        ccl_global Ray *Ray_coop,             /* Required for setting up ray for next iteration */
        ccl_global PathState *PathState_coop, /* Required for setting up ray for next iteration */
        ccl_global Ray *LightRay_dl_coop,     /* Required for radiance update - direct lighting */
        ccl_global int *ISLamp_coop,          /* Required for radiance update - direct lighting */
        ccl_global BsdfEval *BSDFEval_coop,   /* Required for radiance update - direct lighting */
        ccl_global Ray *LightRay_ao_coop,     /* Required for radiance update - AO */
        ccl_global float3 *AOBSDF_coop,       /* Required for radiance update - AO */
        ccl_global float3 *AOAlpha_coop,      /* Required for radiance update - AO */
        ccl_global char *ray_state,           /* Denotes the state of each ray */
        ccl_global int *Queue_data,           /* Queue memory */
        ccl_global int *Queue_index,          /* Tracks the number of elements in each queue */
        int queuesize,                        /* Size (capacity) of each queue */
        ccl_global char *use_queues_flag)     /* flag to decide if scene_intersect kernel should
                                               * use queues to fetch ray index */
{
	ccl_local unsigned int local_queue_atomics;
	if(get_local_id(0) == 0 && get_local_id(1) == 0) {
		local_queue_atomics = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if(get_global_id(0) == 0 && get_global_id(1) == 0) {
		/* If we are here, then it means that scene-intersect kernel
		* has already been executed atleast once. From the next time,
		* scene-intersect kernel may operate on queues to fetch ray index
		*/
		use_queues_flag[0] = 1;

		/* Mark queue indices of QUEUE_SHADOW_RAY_CAST_AO_RAYS and
		 * QUEUE_SHADOW_RAY_CAST_DL_RAYS queues that were made empty during the
		 * previous kernel.
		 */
		Queue_index[QUEUE_SHADOW_RAY_CAST_AO_RAYS] = 0;
		Queue_index[QUEUE_SHADOW_RAY_CAST_DL_RAYS] = 0;
	}

	char enqueue_flag = 0;
	int ray_index = get_global_id(1) * get_global_size(0) + get_global_id(0);
	ray_index = get_ray_index(ray_index,
	                          QUEUE_ACTIVE_AND_REGENERATED_RAYS,
	                          Queue_data,
	                          queuesize,
	                          0);

#ifdef __COMPUTE_DEVICE_GPU__
	/* If we are executing on a GPU device, we exit all threads that are not
	 * required.
	 *
	 * If we are executing on a CPU device, then we need to keep all threads
	 * active since we have barrier() calls later in the kernel. CPU devices,
	 * expect all threads to execute barrier statement.
	 */
	if(ray_index == QUEUE_EMPTY_SLOT) {
		return;
	}
#endif

#ifndef __COMPUTE_DEVICE_GPU__
	if(ray_index != QUEUE_EMPTY_SLOT) {
#endif
		enqueue_flag = kernel_next_iteration_setup((KernelGlobals *)kg,
		                                           SD_REF((ShaderData *)sd, ray_index),
		                                           rng_coop,
		                                           throughput_coop,
		                                           PathRadiance_coop,
		                                           Ray_coop,
		                                           PathState_coop,
		                                           LightRay_dl_coop,
		                                           ISLamp_coop,
		                                           BSDFEval_coop,
		                                           LightRay_ao_coop,
		                                           AOBSDF_coop,
		                                           AOAlpha_coop,
		                                           ray_state,
		                                           use_queues_flag,
		                                           ray_index);
#ifndef __COMPUTE_DEVICE_GPU__
	}
#endif

	/* Enqueue RAY_UPDATE_BUFFER rays. */
	enqueue_ray_index_local(ray_index,
	                        QUEUE_HITBG_BUFF_UPDATE_TOREGEN_RAYS,
	                        enqueue_flag,
	                        queuesize,
	                        &local_queue_atomics,
	                        Queue_data,
	                        Queue_index);
}
/*
 * Copyright 2011-2015 Blender Foundation
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

#include "split/kernel_indirect_subsurface.h"

__kernel void kernel_ocl_path_trace_indirect_subsurface(
        ccl_global char *kg,
        ccl_constant KernelData *data,
        ccl_global SubsurfaceIndirectRays *SSS_coop,
        PathRadiance *PathRadiance_coop,       /* Required for background hit processing and buffer Update */
        ccl_global PathState *PathState_coop,  /* Required for background hit processing */
        ccl_global Ray *Ray_coop,              /* Required for background hit processing */
        ccl_global float3 *throughput_coop,    /* Required for background hit processing */
        ccl_global char *ray_state,            /* Stores information on the current state of a ray */
        ccl_global int *Queue_data,            /* Queues memory */
        ccl_global int *Queue_index,           /* Tracks the number of elements in each queue */
        int queuesize)                         /* Size (capacity) of each queue */
{

	int thread_index = get_global_id(1) * get_global_size(0) + get_global_id(0);
	if(thread_index == 0) {
		/* We will empty both queues in this kernel. */
		Queue_index[QUEUE_ACTIVE_AND_REGENERATED_RAYS] = 0;
		Queue_index[QUEUE_HITBG_BUFF_UPDATE_TOREGEN_RAYS] = 0;
	}

	int ray_index;
	ray_index = get_ray_index(thread_index,
	                          QUEUE_ACTIVE_AND_REGENERATED_RAYS,
	                          Queue_data,
	                          queuesize,
	                          1);
	ray_index = get_ray_index(thread_index,
	                          QUEUE_HITBG_BUFF_UPDATE_TOREGEN_RAYS,
	                          Queue_data,
	                          queuesize,
	                          1);

#ifdef __SUBSURFACE__
	if(ray_index != QUEUE_EMPTY_SLOT) {
		kernel_indirect_subsurface((KernelGlobals *)kg,
		                           SSS_coop,
		                           PathRadiance_coop,
		                           PathState_coop,
		                           Ray_coop,
		                           throughput_coop,
		                           ray_state,
		                           ray_index);
	}
#endif  /* __SUBSURFACE__ */

}
/*
 * Copyright 2011-2015 Blender Foundation
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

#include "split/kernel_buffer_update.h"

__kernel void kernel_ocl_path_trace_buffer_update(
        ccl_global char *kg,
        ccl_constant KernelData *data,
        ccl_global float *per_sample_output_buffers,
        ccl_global uint *rng_state,
        ccl_global uint *rng_coop,             /* Required for buffer Update */
        ccl_global float3 *throughput_coop,    /* Required for background hit processing */
        PathRadiance *PathRadiance_coop,       /* Required for background hit processing and buffer Update */
        ccl_global Ray *Ray_coop,              /* Required for background hit processing */
        ccl_global PathState *PathState_coop,  /* Required for background hit processing */
        ccl_global float *L_transparent_coop,  /* Required for background hit processing and buffer Update */
        ccl_global SubsurfaceIndirectRays *SSS_coop,     /* Required for radiance update - direct lighting */
        ccl_global char *ray_state,            /* Stores information on the current state of a ray */
        int sw, int sh, int sx, int sy, int stride,
        int rng_state_offset_x,
        int rng_state_offset_y,
        int rng_state_stride,
        ccl_global unsigned int *work_array,   /* Denotes work of each ray */
        ccl_global int *Queue_data,            /* Queues memory */
        ccl_global int *Queue_index,           /* Tracks the number of elements in each queue */
        int queuesize,                         /* Size (capacity) of each queue */
        int end_sample,
        int start_sample,
#ifdef __WORK_STEALING__
        ccl_global unsigned int *work_pool_wgs,
        unsigned int num_samples,
#endif
#ifdef __KERNEL_DEBUG__
        DebugData *debugdata_coop,
#endif
        int parallel_samples)                  /* Number of samples to be processed in parallel */
{
	ccl_local unsigned int local_queue_atomics;
	if(get_local_id(0) == 0 && get_local_id(1) == 0) {
		local_queue_atomics = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	int ray_index = get_global_id(1) * get_global_size(0) + get_global_id(0);
	if(ray_index == 0) {
		/* We will empty this queue in this kernel. */
		Queue_index[QUEUE_HITBG_BUFF_UPDATE_TOREGEN_RAYS] = 0;
	}
	char enqueue_flag = 0;
	ray_index = get_ray_index(ray_index,
	                          QUEUE_HITBG_BUFF_UPDATE_TOREGEN_RAYS,
	                          Queue_data,
	                          queuesize,
	                          1);

#ifdef __COMPUTE_DEVICE_GPU__
	/* If we are executing on a GPU device, we exit all threads that are not
	 * required.
	 *
	 * If we are executing on a CPU device, then we need to keep all threads
	 * active since we have barrier() calls later in the kernel. CPU devices,
	 * expect all threads to execute barrier statement.
	 */
	if(ray_index == QUEUE_EMPTY_SLOT) {
		return;
	}
#endif

#ifndef __COMPUTE_DEVICE_GPU__
	if(ray_index != QUEUE_EMPTY_SLOT) {
#endif
		enqueue_flag =
			kernel_buffer_update((KernelGlobals *)kg,
			                     per_sample_output_buffers,
			                     rng_state,
			                     rng_coop,
			                     throughput_coop,
			                     PathRadiance_coop,
			                     Ray_coop,
			                     PathState_coop,
			                     L_transparent_coop,
			                     SSS_coop,
			                     ray_state,
			                     sw, sh, sx, sy, stride,
			                     rng_state_offset_x,
			                     rng_state_offset_y,
			                     rng_state_stride,
			                     work_array,
			                     end_sample,
			                     start_sample,
#ifdef __WORK_STEALING__
			                     work_pool_wgs,
			                     num_samples,
#endif
#ifdef __KERNEL_DEBUG__
			                     debugdata_coop,
#endif
			                     parallel_samples,
			                     ray_index);
#ifndef __COMPUTE_DEVICE_GPU__
	}
#endif

	/* Enqueue RAY_REGENERATED rays into QUEUE_ACTIVE_AND_REGENERATED_RAYS;
	 * These rays will be made active during next SceneIntersectkernel.
	 */
	enqueue_ray_index_local(ray_index,
	                        QUEUE_ACTIVE_AND_REGENERATED_RAYS,
	                        enqueue_flag,
	                        queuesize,
	                        &local_queue_atomics,
	                        Queue_data,
	                        Queue_index);
}
/*
 * Copyright 2011-2015 Blender Foundation
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

//#include "split/kernel_sum_all_radiance.h"
/* Since we process various samples in parallel; The output radiance of different samples
 * are stored in different locations; This kernel combines the output radiance contributed
 * by all different samples and stores them in the RenderTile's output buffer.
 */
ccl_device void kernel_sum_all_radiance(
        ccl_constant KernelData *data,               /* To get pass_stride to offet into buffer */
        ccl_global float *buffer,                    /* Output buffer of RenderTile */
        ccl_global float *per_sample_output_buffer,  /* Radiance contributed by all samples */
        int parallel_samples, int sw, int sh, int stride,
        int buffer_offset_x,
        int buffer_offset_y,
        int buffer_stride,
        int start_sample)
{
	int x = get_global_id(0);
	int y = get_global_id(1);

	if(x < sw && y < sh) {
		buffer += ((buffer_offset_x + x) + (buffer_offset_y + y) * buffer_stride) * (data->film.pass_stride);
		per_sample_output_buffer += ((x + y * stride) * parallel_samples) * (data->film.pass_stride);

		int sample_stride = (data->film.pass_stride);

		int sample_iterator = 0;
		int pass_stride_iterator = 0;
		int num_floats = data->film.pass_stride;

		for(sample_iterator = 0; sample_iterator < parallel_samples; sample_iterator++) {
			for(pass_stride_iterator = 0; pass_stride_iterator < num_floats; pass_stride_iterator++) {
				*(buffer + pass_stride_iterator) =
				        (start_sample == 0 && sample_iterator == 0)
				                ? *(per_sample_output_buffer + pass_stride_iterator)
				                : *(buffer + pass_stride_iterator) + *(per_sample_output_buffer + pass_stride_iterator);
			}
			per_sample_output_buffer += sample_stride;
		}
	}
}

__kernel void kernel_ocl_path_trace_sum_all_radiance(
        ccl_constant KernelData *data,               /* To get pass_stride to offet into buffer */
        ccl_global float *buffer,                    /* Output buffer of RenderTile */
        ccl_global float *per_sample_output_buffer,  /* Radiance contributed by all samples */
        int parallel_samples, int sw, int sh, int stride,
        int buffer_offset_x,
        int buffer_offset_y,
        int buffer_stride,
        int start_sample)
{
	kernel_sum_all_radiance(data,
	                        buffer,
	                        per_sample_output_buffer,
	                        parallel_samples,
	                        sw, sh, stride,
	                        buffer_offset_x,
	                        buffer_offset_y,
	                        buffer_stride,
	                        start_sample);
}
