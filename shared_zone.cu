/**
 * Nearest neighbor search
 * マップ内に、店、工場などのゾーンがある確率で配備されている時、
 * 住宅ゾーンから直近の店、工場までのマンハッタン距離を計算する。
 *
 * 各店、工場から周辺に再帰的に距離を更新していくので、O(N)で済む。
 * しかも、GPUで並列化することで、さらに計算時間を短縮できる。
 *
 * shared memoryを使用して高速化できるか？
 */

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <list>
#include <time.h>

#define CITY_SIZE 10 //200
#define GPU_BLOCK_SIZE 10//40
#define GPU_NUM_THREADS 8 //96
#define GPU_BLOCK_SCALE (1.0)//(1.1)
#define NUM_FEATURES 1
#define QUEUE_MAX 1999
#define MAX_DIST 99
#define HISTORY_SIZE 10000

#define CUDA_CALL(x) {if((x) != cudaSuccess){ \
  printf("CUDA error at %s:%d\n",__FILE__,__LINE__); \
  printf("  %s\n", cudaGetErrorString(cudaGetLastError())); \
  exit(EXIT_FAILURE);}} 


struct ZoneType {
	int type;
	int level;
};

struct ZoningPlan {
	ZoneType zones[CITY_SIZE][CITY_SIZE];
};

struct DistanceMap {
	int distances[CITY_SIZE][CITY_SIZE][NUM_FEATURES];
};

struct Point2D {
	int x;
	int y;

	__host__ __device__
	Point2D() : x(0), y(0) {}

	__host__ __device__
	Point2D(int x, int y) : x(x), y(y) {}
};

__host__ __device__
unsigned int rand(unsigned int* randx) {
    *randx = *randx * 1103515245 + 12345;
    return (*randx)&2147483647;
}

__host__ __device__
float randf(unsigned int* randx) {
	return rand(randx) / (float(2147483647) + 1);
}

__host__ __device__
float randf(unsigned int* randx, float a, float b) {
	return randf(randx) * (b - a) + a;
}

__host__ __device__
int sampleFromCdf(unsigned int* randx, float* cdf, int num) {
	float rnd = randf(randx, 0, cdf[num-1]);

	for (int i = 0; i < num; ++i) {
		if (rnd <= cdf[i]) return i;
	}

	return num - 1;
}

__host__ __device__
int sampleFromPdf(unsigned int* randx, float* pdf, int num) {
	if (num == 0) return 0;

	float cdf[40];
	cdf[0] = pdf[0];
	for (int i = 1; i < num; ++i) {
		if (pdf[i] >= 0) {
			cdf[i] = cdf[i - 1] + pdf[i];
		} else {
			cdf[i] = cdf[i - 1];
		}
	}

	return sampleFromCdf(randx, cdf, num);
}

/**
 * ゾーンプランを生成する。
 */
__host__
void generateZoningPlan(ZoningPlan& zoningPlan, std::vector<float> zoneTypeDistribution) {
	std::vector<float> numRemainings(NUM_FEATURES + 1);
	for (int i = 0; i < NUM_FEATURES + 1; ++i) {
		numRemainings[i] = CITY_SIZE * CITY_SIZE * zoneTypeDistribution[i];
	}

	unsigned int randx = 0;

	for (int r = 0; r < CITY_SIZE; ++r) {
		for (int c = 0; c < CITY_SIZE; ++c) {
			int type = sampleFromPdf(&randx, numRemainings.data(), numRemainings.size());
			zoningPlan.zones[r][c].type = type;
			numRemainings[type] -= 1;
		}
	}
}



/**
 * 直近の店までの距離を計算する（マルチスレッド版、shared memory使用）
 */
__global__
void computeDistanceToStore(ZoningPlan* zoningPlan, DistanceMap* distanceMap, uint3* devQueue, int* devQueueStart, int* devQueueEnd, uint3* devHistory) {
	__shared__ int sDist[(int)(GPU_BLOCK_SIZE * GPU_BLOCK_SCALE)][(int)(GPU_BLOCK_SIZE * GPU_BLOCK_SCALE)][NUM_FEATURES];
	__shared__ uint3 sQueue[QUEUE_MAX + 1];
	__shared__ unsigned int queue_begin;
	__shared__ unsigned int queue_end;
	__shared__ unsigned int history_index;

	queue_begin = 0;
	queue_end = 0;
	history_index = 0;
	__syncthreads();

	// queueを初期化
	for (int i = threadIdx.x; i < QUEUE_MAX + 1; i += GPU_NUM_THREADS) {
		sQueue[i] = make_uint3(CITY_SIZE * 100, CITY_SIZE * 100, NUM_FEATURES);
	}

	// global memoryからshared memoryへコピー
	int num_strides = (GPU_BLOCK_SIZE * GPU_BLOCK_SIZE * GPU_BLOCK_SCALE * GPU_BLOCK_SCALE + GPU_NUM_THREADS - 1) / GPU_NUM_THREADS;
	int r0 = blockIdx.y * GPU_BLOCK_SIZE - GPU_BLOCK_SIZE * (GPU_BLOCK_SCALE - 1) * 0.5;
	int c0 = blockIdx.x * GPU_BLOCK_SIZE - GPU_BLOCK_SIZE * (GPU_BLOCK_SCALE - 1) * 0.5;
	for (int i = 0; i < num_strides; ++i) {
		int r1 = (i * GPU_NUM_THREADS + threadIdx.x) / (int)(GPU_BLOCK_SIZE * GPU_BLOCK_SCALE);
		int c1 = (i * GPU_NUM_THREADS + threadIdx.x) % (int)(GPU_BLOCK_SIZE * GPU_BLOCK_SCALE);

		// これ、忘れてた！！
		if (r1 >= GPU_BLOCK_SIZE * GPU_BLOCK_SCALE || c1 >= GPU_BLOCK_SIZE * GPU_BLOCK_SCALE) continue;

		int type = 0;
		if (r0 + r1 >= 0 && r0 + r1 < CITY_SIZE && c0 + c1 >= 0 && c0 + c1 < CITY_SIZE) {
			type = zoningPlan->zones[r0 + r1][c0 + c1].type;
		}
		for (int feature_id = 0; feature_id < NUM_FEATURES; ++feature_id) {
			distanceMap->distances[r0 + r1][c0 + c1][feature_id] = MAX_DIST;
			if (type - 1 == feature_id) {
				sDist[r1][c1][feature_id] = 0;
				unsigned int q_index = atomicInc(&queue_end, QUEUE_MAX);
				sQueue[q_index] = make_uint3(c1, r1, feature_id);
			} else {
				sDist[r1][c1][feature_id] = MAX_DIST;
			}
		}
	}
	__syncthreads();


	// 距離マップを生成
	unsigned int q_index;
	//while ((q_index = atomicInc(&queue_begin, QUEUE_MAX)) < queue_end) {
	for (int iter = 0; iter < 7000; ++iter) {
		q_index = atomicInc(&queue_begin, QUEUE_MAX);
		
	//while (queue_begin < queue_end) {
		uint3 pt = sQueue[q_index];
		if (pt.x < 0 || pt.x >= GPU_BLOCK_SIZE * GPU_BLOCK_SCALE || pt.y < 0 || pt.y >= GPU_BLOCK_SIZE * GPU_BLOCK_SCALE) {
			break;
		}

		int d = sDist[pt.y][pt.x][pt.z];
		int count = 0;

		if (pt.y > 0) {
			unsigned int old = atomicMin(&sDist[pt.y-1][pt.x][pt.z], d + 1);
			if (old > d + 1) {
				unsigned int q_index2 = atomicInc(&queue_end, QUEUE_MAX);
				sQueue[q_index2] = make_uint3(pt.x, pt.y-1, pt.z);
				count++;
				if (count == 60) {
					unsigned int h_ind = atomicInc(&history_index, HISTORY_SIZE - 1);
					//devHistory[h_ind] = make_uint3(pt.x + pt.y * CITY_SIZE, pt.x + (pt.y-1) * CITY_SIZE, pt.z);
					devHistory[h_ind] = make_uint3(pt.x + pt.y * CITY_SIZE, pt.x + (pt.y-1) * CITY_SIZE, q_index2);
				}
			}
		}
		if (pt.y < CITY_SIZE - 1) {
			unsigned int old = atomicMin(&sDist[pt.y+1][pt.x][pt.z], d + 1);
			if (old > d + 1) {
				unsigned int q_index2 = atomicInc(&queue_end, QUEUE_MAX);
				sQueue[q_index2] = make_uint3(pt.x, pt.y+1, pt.z);
				count++;
				if (count == 60) {
					unsigned int h_ind = atomicInc(&history_index, HISTORY_SIZE - 1);
					//devHistory[h_ind] = make_uint3(pt.x + pt.y * CITY_SIZE, pt.x + (pt.y+1) * CITY_SIZE, pt.z);
					devHistory[h_ind] = make_uint3(pt.x + pt.y * CITY_SIZE, pt.x + (pt.y+1) * CITY_SIZE, q_index2);
				}
			}
		}
		if (pt.x > 0) {
			unsigned int old = atomicMin(&sDist[pt.y][pt.x-1][pt.z], d + 1);
			if (old > d + 1) {
				unsigned int q_index2 = atomicInc(&queue_end, QUEUE_MAX);
				sQueue[q_index2] = make_uint3(pt.x-1, pt.y, pt.z);
				count++;
				if (count == 60) {
					unsigned int h_ind = atomicInc(&history_index, HISTORY_SIZE - 1);
					//devHistory[h_ind] = make_uint3(pt.x + pt.y * CITY_SIZE, pt.x-1 + pt.y * CITY_SIZE, pt.z);
					devHistory[h_ind] = make_uint3(pt.x + pt.y * CITY_SIZE, pt.x-1 + pt.y * CITY_SIZE, q_index2);
				}
			}
		}
		if (pt.x < CITY_SIZE - 1) {
			unsigned int old = atomicMin(&sDist[pt.y][pt.x+1][pt.z], d + 1);
			if (old > d + 1) {
				unsigned int q_index2 = atomicInc(&queue_end, QUEUE_MAX);
				sQueue[q_index2] = make_uint3(pt.x+1, pt.y, pt.z);
				count++;
				if (count == 60) {
					unsigned int h_ind = atomicInc(&history_index, HISTORY_SIZE - 1);
					//devHistory[h_ind] = make_uint3(pt.x + pt.y * CITY_SIZE, pt.x+1 + pt.y * CITY_SIZE, pt.z);
					devHistory[h_ind] = make_uint3(pt.x + pt.y * CITY_SIZE, pt.x+1 + pt.y * CITY_SIZE, q_index2);
				}
			}
		}

		sQueue[q_index] = make_uint3(CITY_SIZE * 100, CITY_SIZE * 100, NUM_FEATURES);
	}
	
	__syncthreads();

	// global memoryの距離マップへ、コピーする
	for (int i = 0; i < num_strides; ++i) {
		int r1 = (i * GPU_NUM_THREADS + threadIdx.x) / (int)(GPU_BLOCK_SIZE * GPU_BLOCK_SCALE);
		int c1 = (i * GPU_NUM_THREADS + threadIdx.x) % (int)(GPU_BLOCK_SIZE * GPU_BLOCK_SCALE);

		// これ、忘れてた！！
		if (r1 >= GPU_BLOCK_SIZE * GPU_BLOCK_SCALE || c1 >= GPU_BLOCK_SIZE * GPU_BLOCK_SCALE) continue;

		if (r0 + r1 >= 0 && r0 + r1 < CITY_SIZE && c0 + c1 >= 0 && c0 + c1 < CITY_SIZE) {
			for (int feature_id = 0; feature_id < NUM_FEATURES; ++feature_id) {
				atomicMin(&distanceMap->distances[r0 + r1][c0 + c1][feature_id], sDist[r1][c1][feature_id]);
			}
		}
	}

	// デバッグ用に、キュー情報をglobalにコピー
	for (int i = 0; i < queue_end; ++i) {
		devQueue[i] = sQueue[i];
	}
	*devQueueStart = queue_begin;
	*devQueueEnd = queue_end;

}

/**
 * 直近の店までの距離を計算する（CPU版）
 */
__host__
void computeDistanceToStoreCPU(ZoningPlan* zoningPLan, DistanceMap* distanceMap) {
	std::list<int3> queue;

	for (int feature_id = 0; feature_id < NUM_FEATURES; ++feature_id) {
		for (int cell_id = 0; cell_id < CITY_SIZE * CITY_SIZE; ++cell_id) {
			int r = cell_id / CITY_SIZE;
			int c = cell_id % CITY_SIZE;

			if (zoningPLan->zones[r][c].type - 1 == feature_id) {
				queue.push_back(make_int3(c, r, feature_id));
				distanceMap->distances[r][c][feature_id] = 0;
			} else {
				distanceMap->distances[r][c][feature_id] = MAX_DIST;
			}
		}
	}

	while (!queue.empty()) {
		int3 pt = queue.front();
		queue.pop_front();

		int d = distanceMap->distances[pt.y][pt.x][pt.z];

		if (pt.y > 0) {
			if (distanceMap->distances[pt.y-1][pt.x][pt.z] > d + 1) {
				distanceMap->distances[pt.y-1][pt.x][pt.z] = d + 1;
				queue.push_back(make_int3(pt.x, pt.y-1, pt.z));
			}
		}
		if (pt.y < CITY_SIZE - 1) {
			if (distanceMap->distances[pt.y+1][pt.x][pt.z] > d + 1) {
				distanceMap->distances[pt.y+1][pt.x][pt.z] = d + 1;
				queue.push_back(make_int3(pt.x, pt.y+1, pt.z));
			}
		}
		if (pt.x > 0) {
			if (distanceMap->distances[pt.y][pt.x-1][pt.z] > d + 1) {
				distanceMap->distances[pt.y][pt.x-1][pt.z] = d + 1;
				queue.push_back(make_int3(pt.x-1, pt.y, pt.z));
			}
		}
		if (pt.x < CITY_SIZE - 1) {
			if (distanceMap->distances[pt.y][pt.x+1][pt.z] > d + 1) {
				distanceMap->distances[pt.y][pt.x+1][pt.z] = d + 1;
				queue.push_back(make_int3(pt.x+1, pt.y, pt.z));
			}
		}
	}
}

/**
 * デバッグ用に、ゾーンプランを表示する。
 */
__host__
void showDevZoningPlan(ZoningPlan* zoningPlan) {
	ZoningPlan plan;

	CUDA_CALL(cudaMemcpy(&plan, zoningPlan, sizeof(ZoningPlan), cudaMemcpyDeviceToHost));
	printf("Zone plan:\n");
	for (int r = CITY_SIZE - 1; r >= 0; --r) {
		for (int c = 0; c < CITY_SIZE; ++c) {
			printf("%2d, ", plan.zones[r][c].type);
		}
		printf("\n");
	}
	printf("\n");
}

/**
 * デバッグ用に、距離マップを表示する。
 */
__host__
void showDevDistMap(DistanceMap* distMap, int feature_id) {
	DistanceMap map;

	CUDA_CALL(cudaMemcpy(&map, distMap, sizeof(DistanceMap), cudaMemcpyDeviceToHost));
	printf("Distance map:\n");
	for (int r = CITY_SIZE - 1; r >= 0; --r) {
		for (int c = 0; c < CITY_SIZE; ++c) {
			printf("%2d, ", map.distances[r][c][feature_id]);
		}
		printf("\n");
	}
	printf("\n");
}

/**
 * デバッグ用に、キューの内容を表示する。
 */
__host__
void showDevQueue(uint3* queue, int* queueBegin, int* queueEnd, int featureId) {
	uint3 q[QUEUE_MAX + 1];
	int begin;
	int end;

	CUDA_CALL(cudaMemcpy(q, queue, sizeof(uint3) * (QUEUE_MAX + 1), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(&begin, queueBegin, sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(&end, queueEnd, sizeof(int), cudaMemcpyDeviceToHost));

	printf("Queue:\n");
	printf("Begin: %d, End: %d, featureId: %d\n", begin, end, featureId);

	int num = end - begin + 1;
	if (num < 0) {
		num = end + QUEUE_MAX + 1 - begin + 1;
	}

	for (int i = 0; i < num; ++i) {
		if (q[(begin + i) % (QUEUE_MAX + 1)].z == featureId) {
			printf("%3d: %3d, %3d\n", (begin + i) % (QUEUE_MAX + 1), q[(begin + i) % (QUEUE_MAX + 1)].x, q[(begin + i) % (QUEUE_MAX + 1)].y);
		}
	}
	printf("\n");
}

__host__
void showDevHistory(uint3* devHistory) {
	uint3 history[HISTORY_SIZE];
	CUDA_CALL(cudaMemcpy(history, devHistory, sizeof(uint3) * HISTORY_SIZE, cudaMemcpyDeviceToHost));

	printf("History:\n");
	for (int i = 0; i < 100; ++i) {
		int x0 = history[i].x % CITY_SIZE;
		int y0 = history[i].x / CITY_SIZE;
		int x1 = history[i].y % CITY_SIZE;
		int y1 = history[i].y / CITY_SIZE;
		int featureId = history[i].z;
		printf("%2d: (%3d, %3d) -> (%3d, %3d) (%d)\n", i, x0, y0, x1, y1, featureId);
	}
	printf("\n");
}


int main()
{
	time_t start, end;


	ZoningPlan* hostZoningPlan = (ZoningPlan*)malloc(sizeof(ZoningPlan));
	DistanceMap* hostDistanceMap = (DistanceMap*)malloc(sizeof(DistanceMap));
	DistanceMap* hostDistanceMap2 = (DistanceMap*)malloc(sizeof(DistanceMap));

	// 距離を初期化
	//memset(hostDistanceMap, MAX_DIST, sizeof(DistanceMap));
	//memset(hostDistanceMap2, MAX_DIST, sizeof(DistanceMap));

	std::vector<float> zoneTypeDistribution(6);
	zoneTypeDistribution[0] = 0.5f;
	zoneTypeDistribution[1] = 0.2f;
	zoneTypeDistribution[2] = 0.1f;
	zoneTypeDistribution[3] = 0.1f;
	zoneTypeDistribution[4] = 0.05f;
	zoneTypeDistribution[5] = 0.05f;
	
	// 初期プランを生成
	start = clock();
	generateZoningPlan(*hostZoningPlan, zoneTypeDistribution);
	end = clock();
	printf("generateZoningPlan: %lf\n", (double)(end-start)/CLOCKS_PER_SEC);
	
	// 初期プランをデバイスバッファへコピー
	ZoningPlan* devZoningPlan;
	CUDA_CALL(cudaMalloc((void**)&devZoningPlan, sizeof(ZoningPlan)));
	CUDA_CALL(cudaMemcpy(devZoningPlan, hostZoningPlan, sizeof(ZoningPlan), cudaMemcpyHostToDevice));

	// デバッグ用に、初期プランを表示
	if (CITY_SIZE <= 100) {
		showDevZoningPlan(devZoningPlan);
	}

	// 距離マップ用に、デバイスバッファを確保
	DistanceMap* devDistanceMap;
	CUDA_CALL(cudaMalloc((void**)&devDistanceMap, sizeof(DistanceMap)));

	///////////////////////////////////////////////////////////////////////
	// CPU版で、直近の店までの距離を計算
	/*
	start = clock();
	for (int iter = 0; iter < 1000; ++iter) {
		computeDistanceToStoreCPU(hostZoningPlan, hostDistanceMap2);
	}
	end = clock();
	printf("computeDistanceToStore CPU: %lf\n", (double)(end-start)/CLOCKS_PER_SEC);
	*/

	int* devQueueStart;
	CUDA_CALL(cudaMalloc((void**)&devQueueStart, sizeof(int)));
	int* devQueueEnd;
	CUDA_CALL(cudaMalloc((void**)&devQueueEnd, sizeof(int)));

	printf("start...\n");



	// デバッグ用に、globalにqueueメモリを確保
	uint3* devQueue;
	CUDA_CALL(cudaMalloc((void**)&devQueue, sizeof(uint3) * (QUEUE_MAX + 1)));

	// デバッグ用に、globalにwavefrontメモリを確保
	uint3* devHistory;
	CUDA_CALL(cudaMalloc((void**)&devHistory, sizeof(uint3) * HISTORY_SIZE));





	/*
	///////////////////////////////////////////////////////////////////////
	// warmp up
	computeDistanceToStore<<<dim3(CITY_SIZE / GPU_BLOCK_SIZE, CITY_SIZE / GPU_BLOCK_SIZE), GPU_NUM_THREADS>>>(devZoningPlan, devDistanceMap, devQueue);

	// マルチスレッドで、直近の店までの距離を計算
	start = clock();
	for (int iter = 0; iter < 1000; ++iter) {
		computeDistanceToStore<<<dim3(CITY_SIZE / GPU_BLOCK_SIZE, CITY_SIZE / GPU_BLOCK_SIZE), GPU_NUM_THREADS>>>(devZoningPlan, devDistanceMap, devQueue);
		cudaDeviceSynchronize();
	}
	end = clock();
	printf("computeDistanceToStore GPU: %lf\n", (double)(end-start)/CLOCKS_PER_SEC);
	*/



	// テンポラリ　デバッグ用
	// バグの対処中。。。。
	FILE* fp = fopen("zone.txt", "r");
	for (int r = 0; r < CITY_SIZE; ++r) {
		for (int c = 0; c < CITY_SIZE; ++c) {
			fscanf(fp, "%d,", &hostZoningPlan->zones[r][c].type);
		}
	}
	CUDA_CALL(cudaMemcpy(devZoningPlan, hostZoningPlan, sizeof(ZoningPlan), cudaMemcpyHostToDevice));
	showDevZoningPlan(devZoningPlan);
	computeDistanceToStore<<<dim3(CITY_SIZE / GPU_BLOCK_SIZE, CITY_SIZE / GPU_BLOCK_SIZE), GPU_NUM_THREADS>>>(devZoningPlan, devDistanceMap, devQueue, devQueueStart, devQueueEnd, devHistory);
	cudaDeviceSynchronize();




	// 距離マップを表示
	showDevDistMap(devDistanceMap, 0);

	// キューを表示
	showDevQueue(devQueue, devQueueStart, devQueueEnd, 0);

	// hisotryを表示
	//showDevHistory(devHistory);

	/*
	// compare the results with the CPU version of the exact algrotihm
	int bad_k = 0;
	int bad_count = 0;
	{	
		for (int r = CITY_SIZE - 1; r >= 0; --r) {
			for (int c = 0; c < CITY_SIZE; ++c) {
				for (int k = 0; k < NUM_FEATURES; ++k) {
					if (hostDistanceMap->distances[r][c][k] != hostDistanceMap2->distances[r][c][k]) {
						if (bad_count == 0) {
							printf("ERROR! %d,%d k=%d, %d != %d\n", r, c, k, hostDistanceMap->distances[r][c][k], hostDistanceMap2->distances[r][c][k]);
							bad_k = k;
						}
						bad_count++;
					}
				}
			}

		}
	}

	// for debug
	if (CITY_SIZE <= 200 && bad_count > 0) {
		for (int r = CITY_SIZE - 1; r >= 0; --r) {
			for (int c = 0; c < CITY_SIZE; ++c) {
				printf("%d, ", hostDistanceMap->distances[r][c][bad_k]);
			}
			printf("\n");
		}
		printf("\n");

		for (int r = CITY_SIZE - 1; r >= 0; --r) {
			for (int c = 0; c < CITY_SIZE; ++c) {
				printf("%d, ", hostDistanceMap2->distances[r][c][bad_k]);
			}
			printf("\n");
		}
		printf("\n");
	}

	printf("Total error: %d\n", bad_count);
	*/

	// release device buffer
	cudaFree(devZoningPlan);
	cudaFree(devDistanceMap);

	// release host buffer
	free(hostZoningPlan);
	free(hostDistanceMap);
	free(hostDistanceMap2);

	//cudaDeviceReset();
}
