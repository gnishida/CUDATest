#include <stdio.h>

#define CITY_SIZE 20
#define BLOCK_SIZE 10
#define NUM_THREADS 10
#define MAX_ITERATIONS 1000

__global__
void test(int* zone, int* dist) {
	// 
	int x0 = blockIdx.x * BLOCK_SIZE;
	int y0 = blockIdx.y * BLOCK_SIZE;

	int stride = ceilf((float)BLOCK_SIZE * BLOCK_SIZE / NUM_THREADS);

	for (int i = threadIdx.x; i < BLOCK_SIZE * BLOCK_SIZE; i += stride) {
		int x = i % BLOCK_SIZE + x0;
		int y = i / BLOCK_SIZE + y0;

		dist[x + y * CITY_SIZE] = zone[x + y * CITY_SIZE];
	}
	dist[0] =99;
	dist[1] = 1;
}

__host__
void showZone(int* devZone) {
	int* zone;
	zone = (int*)malloc(sizeof(int) * CITY_SIZE * CITY_SIZE);
	cudaMemcpy(zone, devZone, sizeof(int) * CITY_SIZE * CITY_SIZE, cudaMemcpyDeviceToHost);

	printf("<<< Zone Map >>>\n");
	for (int r = 0; r < CITY_SIZE; ++r) {
		for (int c = 0; c < CITY_SIZE; ++c) {
			printf("%2d,", zone[r * CITY_SIZE + c]);
		}
		printf("\n");
	}
	printf("\n");

	free(zone);
}

__host__
void showDist(int* devDist) {
	int* dist;
	dist = (int*)malloc(sizeof(int) * CITY_SIZE * CITY_SIZE);
	cudaMemcpy(dist, devDist, sizeof(int) * CITY_SIZE * CITY_SIZE, cudaMemcpyDeviceToHost);

	printf("<<< Distance Map >>>\n");
	for (int r = 0; r < CITY_SIZE; ++r) {
		for (int c = 0; c < CITY_SIZE; ++c) {
			printf("%2d,", dist[r * CITY_SIZE + c]);
		}
		printf("\n");
	}
	printf("\n");

	free(dist);
}

int main(int argc, char **argv) {
	time_t start, end;

	int* hostZone;
	hostZone = (int*)malloc(sizeof(int) * CITY_SIZE * CITY_SIZE);
	for (int i = 0; i < CITY_SIZE * CITY_SIZE; ++i) {
		hostZone[i] = rand() % 6;
	}

	int* devZone;
	cudaMalloc((void**)&devZone, sizeof(int) * CITY_SIZE * CITY_SIZE);
	int* devDist;
	cudaMalloc((void**)&devDist, sizeof(int) * CITY_SIZE * CITY_SIZE);


	cudaMemcpy(devZone, hostZone, sizeof(int) * CITY_SIZE * CITY_SIZE, cudaMemcpyHostToDevice);

	start = clock();
	for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
		test<<<dim3(CITY_SIZE / BLOCK_SIZE, CITY_SIZE / BLOCK_SIZE, 0), NUM_THREADS>>>(devZone, devDist);
		cudaThreadSynchronize();
	}
	end = clock();
	printf("computeDistanceToStore GPU: %lf\n", (double)(end-start)/CLOCKS_PER_SEC);

	showZone(devZone);
	showDist(devDist);
}