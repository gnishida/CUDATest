/**
 * Nearest neighbor search
 * マップ内に店ゾーンが20%の確率で配備されている時、
 * 住宅ゾーンから直近の店ゾーンまでのマンハッタン距離を計算する。
 * Kd-treeなどのアルゴリズムだと、各住宅ゾーンから直近の店までの距離の計算にO(log M)。
 * 従って、全ての住宅ゾーンについて調べると、O(N log M)。
 * 一方、本実装では、各店ゾーンから周辺ゾーンに再帰的に距離を更新していくので、O(N)で済む。
 * しかも、GPUで並列化することで、さらに計算時間を短縮できる。
 */

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <list>
#include <time.h>

#define CITY_SIZE 400
#define NUM_GPU_BLOCKS 4
#define NUM_GPU_THREADS 32
#define NUM_FEATURES 1



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
void generateZoningPlan(ZoningPlan& zoningPlan, std::vector<float> zoneTypeDistribution, std::vector<Point2D>& hostStoreLocations) {
	std::vector<float> numRemainings(zoneTypeDistribution.size());
	for (int i = 0; i < zoneTypeDistribution.size(); ++i) {
		numRemainings[i] = CITY_SIZE * CITY_SIZE * zoneTypeDistribution[i];
	}

	unsigned int randx = 0;

	for (int r = 0; r < CITY_SIZE; ++r) {
		for (int c = 0; c < CITY_SIZE; ++c) {
			int type = sampleFromPdf(&randx, numRemainings.data(), numRemainings.size());
			zoningPlan.zones[r][c].type = type;
			numRemainings[type] -= 1;

			switch (type) {
			case 0:
				break;
			case 1:
				hostStoreLocations.push_back(Point2D(c, r));
				break;
			}
		}
	}
}



/**
 * 直近の店までの距離を計算する
 */
__global__
void computeDistanceToStore(ZoningPlan* zoningPLan, Point2D* storeLocations, int* numStores, DistanceMap* distanceMap) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	Point2D queue[1000];
	int queue_begin = 0;
	int queue_end = 0;

	int stride = ceilf((float)*numStores / NUM_GPU_BLOCKS / NUM_GPU_THREADS);

	for (int i = 0; i < stride; ++i) {
		int store_id = idx * stride + i;
		if (store_id >= *numStores) break;

		queue[queue_end++] = storeLocations[store_id];
		distanceMap->distances[storeLocations[store_id].y][storeLocations[store_id].x][0] = 0;
	}

	while (queue_begin < queue_end) {
		Point2D pt = queue[queue_begin++];

		int d = distanceMap->distances[pt.y][pt.x][0];

		if (pt.y > 0) {
			int old = atomicMin(&distanceMap->distances[pt.y-1][pt.x][0], d + 1);
			if (old > d + 1) {
				queue[queue_end++] = Point2D(pt.x, pt.y-1);
			}
		}
		if (pt.y < CITY_SIZE - 1) {
			int old = atomicMin(&distanceMap->distances[pt.y+1][pt.x][0], d + 1);
			if (old > d + 1) {
				queue[queue_end++] = Point2D(pt.x, pt.y+1);
			}
		}
		if (pt.x > 0) {
			int old = atomicMin(&distanceMap->distances[pt.y][pt.x-1][0], d + 1);
			if (old > d + 1) {
				queue[queue_end++] = Point2D(pt.x-1, pt.y);
			}
		}
		if (pt.x < CITY_SIZE - 1) {
			int old = atomicMin(&distanceMap->distances[pt.y][pt.x+1][0], d + 1);
			if (old > d + 1) {
				queue[queue_end++] = Point2D(pt.x+1, pt.y);
			}
		}
	}
}

/**
 * 直近の店までの距離を計算する
 */
__global__
void computeDistanceToStoreBySingleThread(ZoningPlan* zoningPLan, Point2D* storeLocations, int* numStores, DistanceMap* distanceMap) {
	Point2D queue[1000];
	int queue_begin = 0;
	int queue_end = 0;

	for (int i = 0; i < *numStores; ++i) {
		queue[queue_end++] = storeLocations[i];
		distanceMap->distances[storeLocations[i].y][storeLocations[i].x][0] = 0;
	}

	while (queue_begin < queue_end) {
		Point2D pt = queue[queue_begin++];

		int d = distanceMap->distances[pt.y][pt.x][0];

		if (pt.y > 0) {
			int old = atomicMin(&distanceMap->distances[pt.y-1][pt.x][0], d + 1);
			if (old > d + 1) {
				queue[queue_end++] = Point2D(pt.x, pt.y-1);
			}
		}
		if (pt.y < CITY_SIZE - 1) {
			int old = atomicMin(&distanceMap->distances[pt.y+1][pt.x][0], d + 1);
			if (old > d + 1) {
				queue[queue_end++] = Point2D(pt.x, pt.y+1);
			}
		}
		if (pt.x > 0) {
			int old = atomicMin(&distanceMap->distances[pt.y][pt.x-1][0], d + 1);
			if (old > d + 1) {
				queue[queue_end++] = Point2D(pt.x-1, pt.y);
			}
		}
		if (pt.x < CITY_SIZE - 1) {
			int old = atomicMin(&distanceMap->distances[pt.y][pt.x+1][0], d + 1);
			if (old > d + 1) {
				queue[queue_end++] = Point2D(pt.x+1, pt.y);
			}
		}
	}
}

int main()
{
	time_t start, end;


	ZoningPlan* hostZoningPlan = (ZoningPlan*)malloc(sizeof(ZoningPlan));
	std::vector<Point2D> hostStoreLocations;
	DistanceMap* hostDistanceMap = (DistanceMap*)malloc(sizeof(DistanceMap));
	DistanceMap* hostDistanceMap2 = (DistanceMap*)malloc(sizeof(DistanceMap));

	// 距離を初期化
	memset(hostDistanceMap, 9999, sizeof(DistanceMap));
	memset(hostDistanceMap2, 9999, sizeof(DistanceMap));

	std::vector<float> zoneTypeDistribution(2);
	zoneTypeDistribution[0] = 0.8f;
	zoneTypeDistribution[1] = 0.2f;
	
	// 初期プランを生成
	// 同時に、店の座標リストを作成
	start = clock();
	generateZoningPlan(*hostZoningPlan, zoneTypeDistribution, hostStoreLocations);
	end = clock();
	printf("generateZoningPlan: %lf\n", (double)(end-start)/CLOCKS_PER_SEC);
	/*
	for (int r = CITY_SIZE - 1; r >= 0; --r) {
		for (int c = 0; c < CITY_SIZE; ++c) {
			printf("%d, ", hostZoningPlan.zones[r][c].type);
		}
		printf("\n");
	}
	printf("\n");
	*/

	// 初期プランをデバイスバッファへコピー
	ZoningPlan* devZoningPlan;
	if (cudaMalloc((void**)&devZoningPlan, sizeof(ZoningPlan)) != cudaSuccess) {
		printf("memory allocation error!\n");
		exit(1);
	}
	if (cudaMemcpy(devZoningPlan, hostZoningPlan, sizeof(ZoningPlan), cudaMemcpyHostToDevice) != cudaSuccess) {
		printf("memory copy error!\n");
		exit(1);
	}

	// 店の数をデバイスバッファへコピー
	int* devNumStores;
	if (cudaMalloc((void**)&devNumStores, sizeof(int)) != cudaSuccess) {
		printf("memory allocation error!\n");
		exit(1);
	}
	int hostNumStores = hostStoreLocations.size();
	if (cudaMemcpy(devNumStores, &hostNumStores, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
		printf("memory copy error!\n");
		exit(1);
	}

	// 店の座標をデバイスバッファへコピー
	Point2D* devStoreLocations;
	if (cudaMalloc((void**)&devStoreLocations, sizeof(Point2D) * hostNumStores) != cudaSuccess) {
		printf("memory allocation error!\n");
		exit(1);
	}
	if (cudaMemcpy(devStoreLocations, hostStoreLocations.data(), sizeof(Point2D) * hostNumStores, cudaMemcpyHostToDevice) != cudaSuccess) {
		printf("memory copy error!\n");
		exit(1);
	}


	// 距離マップ用に、デバイスバッファを確保
	DistanceMap* devDistanceMap;
	cudaMalloc((void**)&devDistanceMap, sizeof(DistanceMap));


	///////////////////////////////////////////////////////////////////////
	// シングルスレッドで、直近の店までの距離を計算

	// 距離をデバイスバッファへコピー
	cudaMemcpy(devDistanceMap, hostDistanceMap2, sizeof(DistanceMap), cudaMemcpyHostToDevice);

	// スコアの直近の店までの距離を計算
	start = clock();
	computeDistanceToStoreBySingleThread<<<1, 1>>>(devZoningPlan, devStoreLocations, devNumStores, devDistanceMap);
	end = clock();
	printf("computeDistanceToStoreBySingleThread: %lf\n", (double)(end-start)/CLOCKS_PER_SEC);

	// 距離をCPUバッファへコピー
	cudaMemcpy(hostDistanceMap2, devDistanceMap, sizeof(DistanceMap), cudaMemcpyDeviceToHost);

	///////////////////////////////////////////////////////////////////////
	// マルチスレッドで、直近の店までの距離を計算

	// 距離をデバイスバッファへコピー
	cudaMemcpy(devDistanceMap, hostDistanceMap, sizeof(DistanceMap), cudaMemcpyHostToDevice);

	// スコアの直近の店までの距離を並列で計算
	start = clock();
	computeDistanceToStore<<<NUM_GPU_BLOCKS, NUM_GPU_THREADS>>>(devZoningPlan, devStoreLocations, devNumStores, devDistanceMap);
	end = clock();
	printf("computeDistanceToStore: %lf\n", (double)(end-start)/CLOCKS_PER_SEC);

	// 距離をCPUバッファへコピー
	cudaMemcpy(hostDistanceMap, devDistanceMap, sizeof(DistanceMap), cudaMemcpyDeviceToHost);


	
	// シングルスレッドとマルチスレッドの結果を比較
	for (int r = CITY_SIZE - 1; r >= 0; --r) {
		for (int c = 0; c < CITY_SIZE; ++c) {
			if (hostDistanceMap->distances[r][c][0] != hostDistanceMap2->distances[r][c][0]) {
				printf("ERROR!\n");
			}
		}
	}
	printf("\n");


	// デバイスバッファの開放
	cudaFree(devZoningPlan);
	cudaFree(devNumStores);
	cudaFree(devStoreLocations);
	cudaFree(devDistanceMap);

	// CPUバッファの開放
	free(hostZoningPlan);
	free(hostDistanceMap);
	free(hostDistanceMap2);

	cudaDeviceReset();
}
