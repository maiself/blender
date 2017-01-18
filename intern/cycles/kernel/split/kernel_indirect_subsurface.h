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

/* Note on kernel_background_buffer_update kernel.
 * This is the fourth kernel in the ray tracing logic, and the third
 * of the path iteration kernels. This kernel takes care of rays that hit
 * the background (sceneintersect kernel), and for the rays of
 * state RAY_UPDATE_BUFFER it updates the ray's accumulated radiance in
 * the output buffer. This kernel also takes care of rays that have been determined
 * to-be-regenerated.
 *
 * We will empty QUEUE_HITBG_BUFF_UPDATE_TOREGEN_RAYS queue in this kernel
 *
 * Typically all rays that are in state RAY_HIT_BACKGROUND, RAY_UPDATE_BUFFER
 * will be eventually set to RAY_TO_REGENERATE state in this kernel. Finally all rays of ray_state
 * RAY_TO_REGENERATE will be regenerated and put in queue QUEUE_ACTIVE_AND_REGENERATED_RAYS.
 *
 * The input and output are as follows,
 *
 * rng_coop ---------------------------------------------|--- kernel_buffer_update --|--- PathRadiance_coop
 * throughput_coop --------------------------------------|                           |--- L_transparent_coop
 * per_sample_output_buffers ----------------------------|                           |--- per_sample_output_buffers
 * Ray_coop ---------------------------------------------|                           |--- ray_state
 * PathState_coop ---------------------------------------|                           |--- Queue_data (QUEUE_HITBG_BUFF_UPDATE_TOREGEN_RAYS)
 * L_transparent_coop -----------------------------------|                           |--- Queue_data (QUEUE_ACTIVE_AND_REGENERATED_RAYS)
 * ray_state --------------------------------------------|                           |--- Queue_index (QUEUE_HITBG_BUFF_UPDATE_TOREGEN_RAYS)
 * Queue_data (QUEUE_HITBG_BUFF_UPDATE_TOREGEN_RAYS) ----|                           |--- Queue_index (QUEUE_ACTIVE_AND_REGENERATED_RAYS)
 * Queue_index (QUEUE_ACTIVE_AND_REGENERATED_RAYS) ------|                           |--- work_array
 * parallel_samples -------------------------------------|                           |--- PathState_coop
 * end_sample -------------------------------------------|                           |--- throughput_coop
 * kg (globals) -----------------------------------------|                           |--- rng_coop
 * rng_state --------------------------------------------|                           |--- Ray
 * PathRadiance_coop ------------------------------------|                           |
 * sw ---------------------------------------------------|                           |
 * sh ---------------------------------------------------|                           |
 * sx ---------------------------------------------------|                           |
 * sy ---------------------------------------------------|                           |
 * stride -----------------------------------------------|                           |
 * work_array -------------------------------------------|                           |--- work_array
 * queuesize --------------------------------------------|                           |
 * start_sample -----------------------------------------|                           |--- work_pool_wgs
 * work_pool_wgs ----------------------------------------|                           |
 * num_samples ------------------------------------------|                           |
 *
 * note on sd : sd argument is neither an input nor an output for this kernel. It is just filled and consumed here itself.
 * Note on Queues :
 * This kernel fetches rays from QUEUE_HITBG_BUFF_UPDATE_TOREGEN_RAYS queue.
 *
 * State of queues when this kernel is called :
 * At entry,
 * QUEUE_ACTIVE_AND_REGENERATED_RAYS will be filled with RAY_ACTIVE rays
 * QUEUE_HITBG_BUFF_UPDATE_TOREGEN_RAYS will be filled with RAY_UPDATE_BUFFER, RAY_TO_REGENERATE rays
 * At exit,
 * QUEUE_ACTIVE_AND_REGENERATED_RAYS will be filled with RAY_ACTIVE and RAY_REGENERATED rays
 * QUEUE_HITBG_BUFF_UPDATE_TOREGEN_RAYS will be empty
 */
ccl_device void kernel_indirect_subsurface(
        KernelGlobals *kg,
        ccl_global SubsurfaceIndirectRays *SSS_coop,
        PathRadiance *PathRadiance_coop,       /* Required for background hit processing and buffer Update */
        ccl_global PathState *PathState_coop,  /* Required for background hit processing */
        ccl_global Ray *Ray_coop,              /* Required for background hit processing */
        ccl_global float3 *throughput_coop,    /* Required for background hit processing */
        ccl_global char *ray_state,            /* Stores information on the current state of a ray */
        int ray_index)
{

#ifdef __SUBSURFACE__
	if(IS_STATE(ray_state, ray_index, RAY_UPDATE_BUFFER)) {
		ccl_addr_space SubsurfaceIndirectRays *ss_indirect = &SSS_coop[ray_index];
		PathRadiance *L = &PathRadiance_coop[ray_index];
		kernel_path_subsurface_accum_indirect(ss_indirect, L);

		/* Trace indirect subsurface rays by restarting the loop. this uses less
		 * stack memory than invoking kernel_path_indirect.
		 */
		if(ss_indirect->num_rays) {
			kernel_path_subsurface_setup_indirect(kg,
			                                      ss_indirect,
			                                      &PathState_coop[ray_index],
			                                      &Ray_coop[ray_index],
			                                      L,
			                                      &throughput_coop[ray_index]);
			ASSIGN_RAY_STATE(ray_state, ray_index, RAY_REGENERATED);
		}
		else {
			ASSIGN_RAY_STATE(ray_state, ray_index, RAY_UPDATE_BUFFER);
		}
	}
#endif  /* __SUBSURFACE__ */

}
