typedef struct temp {
	char location[14];
	char date[12];
	float value;
} temp_t;

kernel void identity(global const temp_t* input, global temp_t* output) {
	int gID = get_global_id(0);
	
	output[gID] = input[gID];
}

//
//kernel void mean(global const temp* input, global temp* output) {
//	int gID = get_global_id(0);
//	int gSize = get_global_size(0);
//
//	B[id] = A[id];
//
//	barrier(CLK_GLOBAL_MEM_FENCE);
//
//	for (int i = 1; i < N; i *= 2) { 
//		if (!(id % (i * 2)) && ((id + i) < N))
//			B[id].value += B[id + i].value;
//
//		barrier(CLK_GLOBAL_MEM_FENCE);
//	}
//}