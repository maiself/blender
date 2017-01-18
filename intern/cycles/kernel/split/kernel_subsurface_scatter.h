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

/* Note on kernel_direct_lighting kernel.
 * This is the eighth kernel in the ray tracing logic. This is the seventh
 * of the path iteration kernels. This kernel takes care of direct lighting
 * logic. However, the "shadow ray cast" part of direct lighting is handled
 * in the next kernel.
 *
 * This kernels determines the rays for which a shadow_blocked() function associated with direct lighting should be executed.
 * Those rays for which a shadow_blocked() function for direct-lighting must be executed, are marked with flag RAY_SHADOW_RAY_CAST_DL and
 * enqueued into the queue QUEUE_SHADOW_RAY_CAST_DL_RAYS
 *
 * The input and output are as follows,
 *
 * rng_coop -----------------------------------------|--- kernel_direct_lighting --|--- BSDFEval_coop
 * PathState_coop -----------------------------------|                             |--- ISLamp_coop
 * sd -----------------------------------------------|                             |--- LightRay_coop
 * ray_state ----------------------------------------|                             |--- ray_state
 * Queue_data (QUEUE_ACTIVE_AND_REGENERATED_RAYS) ---|                             |
 * kg (globals) -------------------------------------|                             |
 * queuesize ----------------------------------------|                             |
 *
 * Note on Queues :
 * This kernel only reads from the QUEUE_ACTIVE_AND_REGENERATED_RAYS queue and processes
 * only the rays of state RAY_ACTIVE; If a ray needs to execute the corresponding shadow_blocked
 * part, after direct lighting, the ray is marked with RAY_SHADOW_RAY_CAST_DL flag.
 *
 * State of queues when this kernel is called :
 * state of queues QUEUE_ACTIVE_AND_REGENERATED_RAYS and QUEUE_HITBG_BUFF_UPDATE_TOREGEN_RAYS will be same
 * before and after this kernel call.
 * QUEUE_SHADOW_RAY_CAST_DL_RAYS queue will be filled with rays for which a shadow_blocked function must be executed, after this
 * kernel call. Before this kernel call the QUEUE_SHADOW_RAY_CAST_DL_RAYS will be empty.
 */

#ifdef __SUBSURFACE__

ccl_device char kernel_subsurface_scatter(
        KernelGlobals *kg,
        ShaderData *sd,                              /* Required for direct lighting */
        ccl_global uint *rng_coop,                   /* Required for direct lighting */
        ccl_global PathState *PathState_coop,        /* Required for direct lighting */
        ccl_global Ray *Ray_coop,                    /* Required for direct lighting */
        ccl_global float3 *throughput_coop,          /* Required for direct lighting */
        PathRadiance *PathRadiance_coop,             /* Required for direct lighting */
        ccl_global SubsurfaceIndirectRays *SSS_coop, /* Required for direct lighting */
        ccl_global char *ray_state,                  /* Denotes the state of each ray */
        int ray_index)
{
	if(IS_STATE(ray_state, ray_index, RAY_ACTIVE)) {
		if(ccl_fetch(sd, flag) & SD_BSSRDF) {
			if (kernel_path_subsurface_scatter(kg,
				                               sd,
				                               SD_REF(kg->sd_input, SD_THREAD),
				                               &PathRadiance_coop[ray_index],
				                               &PathState_coop[ray_index],
				                               &rng_coop[ray_index],
				                               &Ray_coop[ray_index],
				                               &throughput_coop[ray_index],
				                               &SSS_coop[ray_index]))
			{
				ASSIGN_RAY_STATE(ray_state, ray_index, RAY_UPDATE_BUFFER);
				return 1;
			}
		}
	}
	return 0;
}

#endif /* __SUBSURFACE__ */

