kernel void identity(global const int* input, global int* output) {
	int gID = get_global_id(0);
	
	output[gID] = input[gID];
}

kernel void sum(global const int* input, global int* output, local int* localSums) {
	int gID = get_global_id(0);
	int lID = get_local_id(0);
	int size = get_local_size(0);

	localSums[lID] = input[gID];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < size; i *= 2) { 
		if (!(lID % (i * 2)) && ((lID + i) < size)) {
			localSums[lID] += localSums[lID + i];
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!lID) {
		atomic_add(&output[0], localSums[lID]);
	}
}