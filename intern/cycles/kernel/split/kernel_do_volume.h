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

/* Note on kernel_lamp_emission
 * This is the 4th kernel in the ray-tracing logic. This is the third of the
 * path-iteration kernels. This kernel takes care of the volume logic.
 * This kernel operates on QUEUE_ACTIVE_AND_REGENERATED_RAYS. It processes rays of state RAY_ACTIVE
 * and RAY_HIT_BACKGROUND.
 * We will empty QUEUE_ACTIVE_AND_REGENERATED_RAYS queue in this kernel.
 * The input/output of the kernel is as follows,
 * Throughput_coop ------------------------------------|--- kernel_lamp_emission --|--- PathRadiance_coop
 * Ray_coop -------------------------------------------|                           |--- Queue_data(QUEUE_ACTIVE_AND_REGENERATED_RAYS)
 * PathState_coop -------------------------------------|                           |--- Queue_index(QUEUE_ACTIVE_AND_REGENERATED_RAYS)
 * kg (globals) ---------------------------------------|                           |
 * Intersection_coop ----------------------------------|                           |
 * ray_state ------------------------------------------|                           |
 * Queue_data (QUEUE_ACTIVE_AND_REGENERATED_RAYS) -----|                           |
 * Queue_index (QUEUE_ACTIVE_AND_REGENERATED_RAYS) ----|                           |
 * queuesize ------------------------------------------|                           |
 * use_queues_flag ------------------------------------|                           |
 * sw -------------------------------------------------|                           |
 * sh -------------------------------------------------|                           |
 */

#ifdef __VOLUME__

ccl_device void kernel_do_volume(
        KernelGlobals *kg,
	ShaderData *sd,
        ccl_global float3 *throughput_coop,    /* Required for volume */
        PathRadiance *PathRadiance_coop,       /* Required for volume */
        ccl_global Ray *Ray_coop,              /* Required for volume */
        ccl_global PathState *PathState_coop,  /* Required for volume */
        Intersection *Intersection_coop,       /* Required for volume */
        ccl_global uint *rng_coop,             /* Required for volume */
        ccl_global char *ray_state,            /* Denotes the state of each ray */
        int sw, int sh,
        ccl_global char *use_queues_flag,      /* Used to decide if this kernel should use
                                                * queues to fetch ray index
                                                */
        int ray_index)
{
	if(IS_STATE(ray_state, ray_index, RAY_ACTIVE) ||
	   IS_STATE(ray_state, ray_index, RAY_HIT_BACKGROUND))
	{
		bool hit = ! IS_FLAG(ray_state, ray_index, RAY_HIT_BACKGROUND);

		PathRadiance *L = &PathRadiance_coop[ray_index];
		ccl_global PathState *state = &PathState_coop[ray_index];

		ccl_global float3 *throughput = &throughput_coop[ray_index];
		ccl_global Ray *ray = &Ray_coop[ray_index];
		ccl_global uint *rng = &rng_coop[ray_index];
		Intersection *isect = &Intersection_coop[ray_index];

		/* Sanitize volume stack. */
		if(!hit) {
			kernel_volume_clean_stack(kg, state->volume_stack);
		}
		/* volume attenuation, emission, scatter */
		if(state->volume_stack[0].shader != SHADER_NONE) {
			Ray volume_ray = *ray;
			volume_ray.t = (hit)? isect->t: FLT_MAX;

			bool heterogeneous = volume_stack_is_heterogeneous(kg, state->volume_stack);

			{
				/* integrate along volume segment with distance sampling */
				VolumeIntegrateResult result = kernel_volume_integrate(
					kg, state, sd, &volume_ray, L, throughput, rng, heterogeneous);

#  ifdef __VOLUME_SCATTER__
				if(result == VOLUME_PATH_SCATTERED) {
					/* direct lighting */
					kernel_path_volume_connect_light(kg, rng, sd, kg->sd_input, *throughput, state, L);

					/* indirect light bounce */
					if(kernel_path_volume_bounce(kg, rng, sd, throughput, state, L, ray))
						ASSIGN_RAY_STATE(ray_state, ray_index, RAY_REGENERATED);
					else
						ASSIGN_RAY_STATE(ray_state, ray_index, RAY_UPDATE_BUFFER);
				}
#  endif
			}
		}
	}
}

#endif /* __VOLUME__ */
