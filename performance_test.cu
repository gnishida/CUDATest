#include <vector>
#include <iostream>

#define GRID_SIZE	200

struct zone_type {
	int type;
	int level;
};

struct zone_plan {
	zone_type zones[GRID_SIZE][GRID_SIZE];
	float score;
};

__host__ __device__
void MCMC(int numIterations) {
	float count = 0.0;

	zone_plan* hoge = (zone_plan*)malloc(sizeof(zone_plan));
	zone_plan* hoge2 = (zone_plan*)malloc(sizeof(zone_plan));
	for (int i = 0; i < numIterations; ++i) {
		memcpy(hoge, hoge2, sizeof(zone_plan));
		for (int r = 0; r < GRID_SIZE; ++r) {
			for (int c = 0; c < GRID_SIZE; ++c) {
			}
		}
	}
}

/**
 * CUDA version of MCMCM
 */
__global__
void MCMCGPUKernel(int* numIterations) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	// initialize random
	unsigned int randx = idx;

	MCMC(*numIterations);
}

/**
 * CUDA version of MCMC
 */
__host__
void zonePlanMCMCGPUfunc(int numIterations) {
	int* devNumIterations;
	if (cudaMalloc((void**)&devNumIterations, sizeof(int)) != cudaSuccess) {
		printf("cuda memory allocation error!\n");
		return;
	}

	if (cudaMemcpy(devNumIterations, &numIterations, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
		printf("cuda memory copy error!\n");
		return;
	}

	// start kernel
	time_t start = clock();
    MCMCGPUKernel<<<1, 1>>>(devNumIterations);
	cudaDeviceSynchronize();
	time_t end = clock();
	printf("Time elapsed: %lf\n", (double)(end-start)/CLOCKS_PER_SEC);
}

void main() {
	zonePlanMCMCGPUfunc(10000);

	// CPU
	time_t start = clock();
	MCMC(10000);
	time_t end = clock();
	printf("CPU version took %lf\n", (double)(end-start)/CLOCKS_PER_SEC);
}