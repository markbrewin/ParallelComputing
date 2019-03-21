//kernel void difFromMean(global const float* input, global float* output, global const float* mean) {
//	int gID = get_global_id(0);
//
//	output[gID] = input[gID] - mean;
//}

//kernel void identity(global const int* input, global int* output) {
//	int gID = get_global_id(0);
//	
//	output[gID] = input[gID];
//}

//kernel void histogram(global const float* input, global int* hist) {
//	int gID = get_global_id(0);
//
//	int bin = round(input[gID] * 100);
//
//	atomic_inc(2500 + &hist[bin]);
//}

void cmpxchg(local float* A, local float* B) {
	if (*A > *B) {
		float t = *A;
		*A = *B;
		*B = t;
	}
}

kernel void minimum(global const float* input, global float* output, local float* mins) {
	int gID = get_global_id(0);
	int groupID = get_group_id(0);
	int lID = get_local_id(0);
	int size = get_local_size(0);

	mins[lID] = input[gID];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < size; i *= 2) {
		if (!(lID % (i * 2)) && ((lID + i) < size)) {
			if (mins[lID] > mins[lID + i] && mins[lID + i] != -999) {
				mins[lID] = mins[lID + i];
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	output[groupID] = mins[0];
}

kernel void maximum(global const float* input, global float* output, local float* maxs) {
	int gID = get_global_id(0);
	int groupID = get_group_id(0);
	int lID = get_local_id(0);
	int size = get_local_size(0);

	maxs[lID] = input[gID];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < size; i *= 2) {
		if (!(lID % (i * 2)) && ((lID + i) < size)) {
			if (maxs[lID] < maxs[lID + i] && maxs[lID + i] != 999) {
				maxs[lID] = maxs[lID + i];
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	output[groupID] = maxs[0];
}

kernel void median(global const float* input, global float* output, global const bool* l, local float* sorted) {
	int gID = get_global_id(0);
	int groupID = get_group_id(0);
	int lID = get_local_id(0);
	int size = get_local_size(0);

	sorted[lID] = input[gID];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < size; i += 2) {
		if (lID % 2 == 0 && lID + 1 < size) {
			cmpxchg(&sorted[lID], &sorted[lID + 1]);
		}

		barrier(CLK_LOCAL_MEM_FENCE);
		
		if (lID % 2 == 1 && lID + 1 < size) {
			cmpxchg(&sorted[lID], &sorted[lID + 1]);
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (l[0] == true) {
		output[groupID] = (sorted[(size / 2) - 1] + sorted[(size / 2)]) / 2;
	}
	else {
		output[groupID] = (sorted[(size / 2) + 1] + sorted[(size / 2)]) / 2;
	}
}

kernel void stddev(global const float* input, global float* output) {
	int gID = get_global_id(0);
	int groupID = get_group_id(0);
	int lID = get_local_id(0);
	int size = get_local_size(0);


}

kernel void sum(global const float* input, global float* output, local float* sums) {
	int gID = get_global_id(0);
	int groupID = get_group_id(0);
	int lID = get_local_id(0);
	int size = get_local_size(0);

	sums[lID] = input[gID];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < size; i *= 2) { 
		if (!(lID % (i * 2)) && ((lID + i) < size)) {
			sums[lID] += sums[lID + i];
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	output[groupID] = sums[0];
}