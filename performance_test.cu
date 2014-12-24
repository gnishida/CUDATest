#include <stdio.h>
#include <vector>

#define CITY_SIZE 200 //200
#define BLOCK_SIZE 40 //200
#define NUM_THREADS 32
#define MAX_ITERATIONS 1000
#define QUEUE_MAX 39999

__global__
void testShared(int* zone, int* dist) {
	__shared__ int sub_zone[BLOCK_SIZE * BLOCK_SIZE];
	__shared__ int sub_dist[BLOCK_SIZE * BLOCK_SIZE];

	// 
	int x0 = blockIdx.x * BLOCK_SIZE;
	int y0 = blockIdx.y * BLOCK_SIZE;

	for (int i = threadIdx.x; i < BLOCK_SIZE * BLOCK_SIZE; i += NUM_THREADS) {
		int x = i % BLOCK_SIZE;
		int y = i / BLOCK_SIZE;

		sub_zone[x + y * CITY_SIZE] = zone[x + x0 + (y + y0) * CITY_SIZE];
	}

	__syncthreads();


	for (int i = threadIdx.x; i < BLOCK_SIZE * BLOCK_SIZE; i += NUM_THREADS) {
		int x = i % BLOCK_SIZE;
		int y = i / BLOCK_SIZE;

		int total = sub_zone[x + y * BLOCK_SIZE];
		int count = 1;
		if (y > 0) {
			total += sub_zone[x + (y - 1) * BLOCK_SIZE];
			count++;
		}
		if (y < BLOCK_SIZE - 1) {
			total += sub_zone[x + (y + 1) * BLOCK_SIZE];
			count++;
		}
		if (x > 0) {
			total += sub_zone[x - 1 + y * BLOCK_SIZE];
			count++;
		}
		if (x < BLOCK_SIZE - 1) {
			total += sub_zone[x + 1 + y * BLOCK_SIZE];
			count++;
		}

		sub_dist[x + y * CITY_SIZE] = total / count;
	}


	__syncthreads();

	for (int i = threadIdx.x; i < BLOCK_SIZE * BLOCK_SIZE; i += NUM_THREADS) {
		int x = i % BLOCK_SIZE;
		int y = i / BLOCK_SIZE;

		dist[x + x0 + (y + y0) * CITY_SIZE] = sub_dist[x + y * CITY_SIZE]; 
	}
}

__global__
void testGlobal(int* zone, int* dist) {
	// 
	int x0 = blockIdx.x * BLOCK_SIZE;
	int y0 = blockIdx.y * BLOCK_SIZE;

	for (int i = threadIdx.x; i < BLOCK_SIZE * BLOCK_SIZE; i += NUM_THREADS) {
		int x = i % BLOCK_SIZE + x0;
		int y = i / BLOCK_SIZE + y0;

		int total = zone[x + y * CITY_SIZE];
		int count = 1;
		if (y > 0) {
			total += zone[x + (y - 1) * CITY_SIZE];
			count++;
		}
		if (y < CITY_SIZE - 1) {
			total += zone[x + (y + 1) * CITY_SIZE];
			count++;
		}
		if (x > 0) {
			total += zone[x - 1 + y * CITY_SIZE];
			count++;
		}
		if (x < CITY_SIZE - 1) {
			total += zone[x + 1 + y * CITY_SIZE];
			count++;
		}

		dist[x + y * CITY_SIZE] = total / count;
	}
}

__host__
void testCPU(int* zone, int* dist) {
	for (int r = 0; r < CITY_SIZE; ++r) {
		for (int c = 0; c < CITY_SIZE; ++c) {
			int total = zone[r * CITY_SIZE + c];
			int count = 1;
			if (r > 0) {
				total += zone[(r - 1) * CITY_SIZE + c];
				count++;
			}
			if (r < CITY_SIZE - 1) {
				total += zone[(r + 1) * CITY_SIZE + c];
				count++;
			}
			if (c > 0) {
				total += zone[r * CITY_SIZE + c - 1];
				count++;
			}
			if (c > CITY_SIZE - 1) {
				total += zone[r * CITY_SIZE + c + 1];
				count++;
			}

			dist[r * CITY_SIZE + c] = total / count;
		}
	}
}

__global__
void testQueueGlobal(int* zone, int* dist, int* queue, unsigned int* q_head, unsigned int* q_tail) {
	int x0 = blockIdx.x * BLOCK_SIZE;
	int y0 = blockIdx.y * BLOCK_SIZE;

	int stride = BLOCK_SIZE * BLOCK_SIZE / NUM_THREADS;
	int x = threadIdx.x * stride % BLOCK_SIZE;
	int y = threadIdx.x * stride / BLOCK_SIZE;

	unsigned int q_index = atomicInc(q_tail, QUEUE_MAX);
	queue[q_index] = x + x0 + (y + y0) * CITY_SIZE;
	dist[x + x0 + (y + y0) * CITY_SIZE] = 0;

	__syncthreads();

	while (true) {
		int q_index = atomicInc(q_head, QUEUE_MAX);
		int s = queue[q_index];
		if (s == -1) break;

		int x = s % CITY_SIZE;
		int y = s / CITY_SIZE;
		int d = dist[s];

		if (x > 0) {
			int old = atomicMin(&dist[s - 1], d + 1);
			if (old > d + 1) {
				unsigned int q_index2 = atomicInc(q_tail, QUEUE_MAX);
				queue[q_index2] = s - 1;
			}
		}
		if (x < CITY_SIZE - 1) {
			int old = atomicMin(&dist[s + 1], d + 1);
			if (old > d + 1) {
				unsigned int q_index2 = atomicInc(q_tail, QUEUE_MAX);
				queue[q_index2] = s + 1;
			}
		}
		if (y > 0) {
			int old = atomicMin(&dist[s - CITY_SIZE], d + 1);
			if (old > d + 1) {
				unsigned int q_index2 = atomicInc(q_tail, QUEUE_MAX);
				queue[q_index2] = s - CITY_SIZE;
			}
		}
		if (y < CITY_SIZE - 1) {
			int old = atomicMin(&dist[s + CITY_SIZE], d + 1);
			if (old > d + 1) {
				unsigned int q_index2 = atomicInc(q_tail, QUEUE_MAX);
				queue[q_index2] = s + CITY_SIZE;
			}
		}

		queue[q_index] = -1;
	}
}

__host__
void testQueueCPU(int* zone, int* dist) {
	std::vector<int> queue(QUEUE_MAX + 1);
	unsigned int q_head = 0;
	unsigned int q_tail = 0;

	for (int i = 0; i < QUEUE_MAX + 1; ++i) queue[i] = -1;

	for (int i = 0; i < 10; ++i) {
		int stride = CITY_SIZE * CITY_SIZE / 10;
		int x = i * stride % CITY_SIZE;
		int y = i * stride / CITY_SIZE;
		
		queue[q_tail++] = x + y * CITY_SIZE;
		dist[x + y * CITY_SIZE] = 0;
	}

	while (true) {
		int s = queue[q_head];
		if (s == -1) break;

		int x = s % CITY_SIZE;
		int y = s / CITY_SIZE;
		int d = dist[s];

		if (x > 0) {
			if (dist[s - 1] > d + 1) {
				dist[s - 1] = d + 1;
				queue[q_tail++] = s - 1;
			}
		}
		if (x < CITY_SIZE - 1) {
			if (dist[s + 1] > d + 1) {
				dist[s + 1] = d + 1;
				queue[q_tail++] = s + 1;
			}
		}
		if (y > 0) {
			if (dist[s - CITY_SIZE] > d + 1) {
				dist[s - CITY_SIZE] = d + 1;
				queue[q_tail++] = s - CITY_SIZE;
			}
		}
		if (y < CITY_SIZE - 1) {
			if (dist[s + CITY_SIZE] > d + 1) {
				dist[s + CITY_SIZE] = d + 1;
				queue[q_tail++] = s + CITY_SIZE;
			}
		}

		queue[q_head++] = -1;
	}
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
	int* hostDist;
	hostDist = (int*)malloc(sizeof(int) * CITY_SIZE * CITY_SIZE);

	int* devZone;
	cudaMalloc((void**)&devZone, sizeof(int) * CITY_SIZE * CITY_SIZE);
	int* devDist;
	cudaMalloc((void**)&devDist, sizeof(int) * CITY_SIZE * CITY_SIZE);


	cudaMemcpy(devZone, hostZone, sizeof(int) * CITY_SIZE * CITY_SIZE, cudaMemcpyHostToDevice);

	start = clock();
	for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
		testGlobal<<<dim3(CITY_SIZE / BLOCK_SIZE, CITY_SIZE / BLOCK_SIZE), NUM_THREADS>>>(devZone, devDist);
		cudaThreadSynchronize();
	}
	end = clock();
	printf("GPU (global): %lf\n", (double)(end-start)/CLOCKS_PER_SEC);

	start = clock();
	for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
		testShared<<<dim3(CITY_SIZE / BLOCK_SIZE, CITY_SIZE / BLOCK_SIZE), NUM_THREADS>>>(devZone, devDist);
		cudaThreadSynchronize();
	}
	end = clock();
	printf("GPU (shared): %lf\n", (double)(end-start)/CLOCKS_PER_SEC);

	start = clock();
	for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
		testCPU(hostZone, hostDist);
	}
	end = clock();
	printf("CPU: %lf\n", (double)(end-start)/CLOCKS_PER_SEC);
	//showZone(devZone);
	//showDist(devDist);



	int* devQueue;
	cudaMalloc((void**)&devQueue, sizeof(int) * (QUEUE_MAX + 1));
	unsigned int* devQueueHead;
	cudaMalloc((void**)&devQueueHead, sizeof(unsigned int));
	unsigned int* devQueueTail;
	cudaMalloc((void**)&devQueueTail, sizeof(unsigned int));
	
	start = clock();
	for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
		cudaMemset(devQueue, -1, sizeof(int) * (QUEUE_MAX + 1));
		cudaMemset(devDist, 99, sizeof(int) * CITY_SIZE * CITY_SIZE);
		cudaMemset(devQueueHead, 0, sizeof(unsigned int));
		cudaMemset(devQueueTail, 0, sizeof(unsigned int));
		testQueueGlobal<<<dim3(CITY_SIZE / BLOCK_SIZE, CITY_SIZE / BLOCK_SIZE), NUM_THREADS>>>(devZone, devDist, devQueue, devQueueHead, devQueueTail);
		cudaThreadSynchronize();
	}
	end = clock();
	printf("GPU (global): %lf\n", (double)(end-start)/CLOCKS_PER_SEC);

	start = clock();
	for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
		cudaMemset(devQueue, -1, sizeof(int) * (QUEUE_MAX + 1));
		cudaMemset(devDist, 99, sizeof(int) * CITY_SIZE * CITY_SIZE);
		cudaMemset(devQueueTail, 0, sizeof(unsigned int));
		testQueueCPU(hostZone, hostDist);
		cudaThreadSynchronize();
	}
	end = clock();
	printf("GPU (global): %lf\n", (double)(end-start)/CLOCKS_PER_SEC);
}