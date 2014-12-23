#include <stdio.h>

#define CITY_SIZE 20
#define BLOCK_SIZE 10
#define NUM_THREADS 10

__global__
void test(int* zone, int* dist) {
	// 
	int x0 = blockIdx.x * BLOCK_SIZE;
	int y0 = blockIdx.y * BLOCK_SIZE;

	int stride = ceilf((float)BLOCK_SIZE * BLOCK_SIZE / NUM_THREADS);

	for (int i = 0; i < BLOCK_SIZE * BLOCK_SIZE; r += stride) {
		int x = i % BLOCK_SIZE + x0;
		int y = i / BLOCK_SIZE + y0;

		dist[x + y * CITY_SIZE] = zone[x + y * CITY_SIZE];
	}
}

int main(int argc, char **argv) {




	test<<<dim(CITY_SIZE / BLOCK_SIZE, CITY_SIZE / BLOCK_SIZE, 0), 10>>>(devZone, devDist);
}