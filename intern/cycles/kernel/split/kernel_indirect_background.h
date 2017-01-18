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

#include "kernel_split_common.h"

/* Note on kernel_indirect_background kernel.
 * This is the fourth kernel in the ray tracing logic, and the third
 * of the path iteration kernels. This kernel takes care of rays that hit
 * the background (sceneintersect kernel)
 *
 * Typically all rays that are in state RAY_HIT_BACKGROUND
 * will be set to RAY_UPDATE_BUFFER state in this kernel.
 *
 * The input and output are as follows,
 *
 * ------------------------------------------------------|--- kernel_indirect_background --|--- PathRadiance_coop
 * throughput_coop --------------------------------------|                                 |--- L_transparent_coop
 * Ray_coop ---------------------------------------------|                                 |--- ray_state
 * PathState_coop ---------------------------------------|                                 |--- Queue_data (QUEUE_HITBG_BUFF_UPDATE_TOREGEN_RAYS)
 * L_transparent_coop -----------------------------------|                                 |--- Queue_data (QUEUE_ACTIVE_AND_REGENERATED_RAYS)
 * ray_state --------------------------------------------|                                 |--- Queue_index (QUEUE_HITBG_BUFF_UPDATE_TOREGEN_RAYS)
 * Queue_data (QUEUE_HITBG_BUFF_UPDATE_TOREGEN_RAYS) ----|                                 |--- Queue_index (QUEUE_ACTIVE_AND_REGENERATED_RAYS)
 * Queue_index (QUEUE_ACTIVE_AND_REGENERATED_RAYS) ------|                                 |--- work_array
 * ------------------------------------------------------|                                 |--- PathState_coop
 * ------------------------------------------------------|                                 |--- throughput_coop
 * kg (globals) -----------------------------------------|                                 |--- rng_coop
 * rng_state --------------------------------------------|                                 |--- Ray
 * PathRadiance_coop ------------------------------------|                                 |
 * queuesize --------------------------------------------|                                 |
 *
 * Note on Queues :
 * This kernel fetches rays from QUEUE_HITBG_BUFF_UPDATE_TOREGEN_RAYS queue.
 *
 * State of queues when this kernel is called :
 * At entry,
 * QUEUE_ACTIVE_AND_REGENERATED_RAYS will be filled with RAY_ACTIVE rays
 * QUEUE_HITBG_BUFF_UPDATE_TOREGEN_RAYS will be filled with RAY_UPDATE_BUFFER, RAY_HIT_BACKGROUND, RAY_TO_REGENERATE rays
 * At exit,
 * QUEUE_ACTIVE_AND_REGENERATED_RAYS will be filled with RAY_ACTIVE rays
 * QUEUE_HITBG_BUFF_UPDATE_TOREGEN_RAYS will be filled with RAY_UPDATE_BUFFER and RAY_TO_REGENERATE rays
 */
ccl_device char kernel_indirect_background(
        KernelGlobals *kg,
        ccl_global float3 *throughput_coop,    /* Required for background hit processing */
        PathRadiance *PathRadiance_coop,       /* Required for background hit processing and buffer Update */
        ccl_global Ray *Ray_coop,              /* Required for background hit processing */
        ccl_global PathState *PathState_coop,  /* Required for background hit processing */
        ccl_global float *L_transparent_coop,  /* Required for background hit processing and buffer Update */
        ccl_global char *ray_state,            /* Stores information on the current state of a ray */
        int ray_index)
{
	char enqueue_flag = 0;

	ccl_global PathState *state = &PathState_coop[ray_index];
	PathRadiance *L = L = &PathRadiance_coop[ray_index];
	ccl_global Ray *ray = &Ray_coop[ray_index];
	ccl_global float3 *throughput = &throughput_coop[ray_index];
	ccl_global float *L_transparent = &L_transparent_coop[ray_index];


	if(IS_STATE(ray_state, ray_index, RAY_HIT_BACKGROUND)) {
		/* eval background shader if nothing hit */
		if(kernel_data.background.transparent && (state->flag & PATH_RAY_CAMERA)) {
			*L_transparent = (*L_transparent) + average((*throughput));
#ifdef __PASSES__
			if(!(kernel_data.film.pass_flag & PASS_BACKGROUND))
#endif
			{
				ASSIGN_RAY_STATE(ray_state, ray_index, RAY_UPDATE_BUFFER);
				enqueue_flag = 1;
			}
		}

		if(IS_STATE(ray_state, ray_index, RAY_HIT_BACKGROUND)) {
#ifdef __BACKGROUND__
			/* sample background shader */
			float3 L_background = indirect_background(kg, SD_REF(kg->sd_input, SD_THREAD), state, ray);
			path_radiance_accum_background(L, (*throughput), L_background, state->bounce);
#endif
			ASSIGN_RAY_STATE(ray_state, ray_index, RAY_UPDATE_BUFFER);
			enqueue_flag = 1;
		}
	}

	return enqueue_flag;
}
