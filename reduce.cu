#include<stdio.h>
#include<math.h>

#define N 1024

__global__ void interleaved_reduce(int* d_in, int* d_out) {
	int i = threadIdx.x;

	__shared__ int sB[N];
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	sB[i] = d_in[id];
	__syncthreads();

	for(int s = 1; s < blockDim.x; s = s*2) {
		int index = 2 * s * id;
		if(index < blockDim.x) {
			sB[index] += sB[index+s];
		}
		__syncthreads();
	}
	if(i == 0)
		d_out[blockIdx.x] = sB[0];
}

__global__ void contiguous_reduce(int* d_in, int* d_out) {
	int i = threadIdx.x;
    int M = N/2;
    for(int s = M; s > 0; s=s>>1) {
        if(i < M) {
            d_in[i] = d_in[i] + d_in[i+s];
        }
        M = M/2;
    }
    if(i == 0)
        d_out[0] = d_in[0];
}


int main() {
	int h_in[N];
	int h_out;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	for(int i = 0; i < N; i++) {
		h_in[i] = i+1;	
	}
	
	int *d_in, *d_out;

	//Part 1: Memory transfer from host to device
	cudaMalloc((void**) &d_in, N*sizeof(int));
	cudaMalloc((void**) &d_out, sizeof(int));

	cudaMemcpy(d_in, &h_in, N*sizeof(int), cudaMemcpyHostToDevice);

	//Part 2: Execute kernel
	
	cudaEventRecord(start);
    // interleaved_reduce<<<1, 1024>>>(d_in, d_out);
    contiguous_reduce<<<1, 1024>>>(d_in, d_out);
	cudaEventRecord(stop);

	//Part 3: Memory transfer from device to host
	cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cudaFree(d_in);
	cudaFree(d_out);

	printf("Output: %d\n", h_out);
	printf("%f milliseconds\n", milliseconds);

	return -1;
}