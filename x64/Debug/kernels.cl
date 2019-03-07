#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

typedef struct temp {
	char location[14];
	char date[12];
	int value;
} temp_t;

kernel void identity(global const int* input, global int* output) {
	int gID = get_global_id(0);
	
	output[gID] = input[gID];
}

kernel void sum(global const int* input, global int* output, local int* localSums) {
	int gID = get_global_id(0);
	int lID = get_local_id(0);
	int size = get_local_size(0);

	//Copies from global to cache
	localSums[lID] = input[gID];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < size; i *= 2) { 
		if (!(lID % (i * 2)) && ((lID + i) < size)) {
			localSums[lID] += localSums[lID + i];
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	output[gID] = localSums[0];
}

kernel void reduce_add_4(global const int* A, global int* B, local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	if (!lid) {
		atomic_add(&B[0],scratch[lid]);
	}
}