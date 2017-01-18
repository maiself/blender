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
