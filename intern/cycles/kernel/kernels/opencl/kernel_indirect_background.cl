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
