/**
 * Nearest neighbor search
 * マップ内に、店、工場などのゾーンがある確率で配備されている時、
 * 住宅ゾーンから直近の店、工場までのマンハッタン距離を計算する。
 *
 * 各店、工場から周辺に再帰的に距離を更新していくので、O(N)で済む。
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
#define NUM_FEATURES 5



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

struct Point2DAndFeature {
	int x;
	int y;
	int featureId;

	__host__ __device__
	Point2DAndFeature() : x(0), y(0), featureId(0) {}

	__host__ __device__
	Point2DAndFeature(int x, int y, int featureId) : x(x), y(y), featureId(featureId) {}
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
		}
	}
}



/**
 * 直近の店までの距離を計算する（マルチスレッド版）
 */
__global__
void computeDistanceToStore(ZoningPlan* zoningPLan, DistanceMap* distanceMap) {
	// キュー
	Point2DAndFeature queue[1000];
	int queue_begin = 0;
	int queue_end = 0;

	int stride_features = ceilf((float)NUM_FEATURES / NUM_GPU_BLOCKS);
	int stride_cells = ceilf((float)(CITY_SIZE * CITY_SIZE) / NUM_GPU_THREADS);
	
	// 分割された領域内で、店を探す
	for (int feature_offset = 0; feature_offset < stride_features; ++feature_offset) {
		int feature_id = blockIdx.x * stride_features + feature_offset;

		for (int cell_offset = 0; cell_offset < stride_cells; ++cell_offset) {
			int r = (threadIdx.x * stride_cells + cell_offset) / CITY_SIZE;
			int c = (threadIdx.x * stride_cells + cell_offset) % CITY_SIZE;

			if (zoningPLan->zones[r][c].type - 1 == feature_id) {
				queue[queue_end++] = Point2DAndFeature(c, r, feature_id);
				distanceMap->distances[r][c][feature_id] = 0;
			}
		}
	}


	// 距離マップを生成
	while (queue_begin < queue_end) {
		Point2DAndFeature pt = queue[queue_begin++];

		int d = distanceMap->distances[pt.y][pt.x][pt.featureId];

		if (pt.y > 0) {
			int old = atomicMin(&distanceMap->distances[pt.y-1][pt.x][pt.featureId], d + 1);
			if (old > d + 1) {
				queue[queue_end++] = Point2DAndFeature(pt.x, pt.y-1, pt.featureId);
			}
		}
		if (pt.y < CITY_SIZE - 1) {
			int old = atomicMin(&distanceMap->distances[pt.y+1][pt.x][pt.featureId], d + 1);
			if (old > d + 1) {
				queue[queue_end++] = Point2DAndFeature(pt.x, pt.y+1, pt.featureId);
			}
		}
		if (pt.x > 0) {
			int old = atomicMin(&distanceMap->distances[pt.y][pt.x-1][pt.featureId], d + 1);
			if (old > d + 1) {
				queue[queue_end++] = Point2DAndFeature(pt.x-1, pt.y, pt.featureId);
			}
		}
		if (pt.x < CITY_SIZE - 1) {
			int old = atomicMin(&distanceMap->distances[pt.y][pt.x+1][pt.featureId], d + 1);
			if (old > d + 1) {
				queue[queue_end++] = Point2DAndFeature(pt.x+1, pt.y, pt.featureId);
			}
		}
	}
}

/**
 * 直近の店までの距離を計算する（シングルスレッド版）
 */
__global__
void computeDistanceToStoreBySingleThread(ZoningPlan* zoningPLan, DistanceMap* distanceMap) {
	// キュー
	Point2DAndFeature queue[1000];
	int queue_begin = 0;
	int queue_end = 0;
	
	// 分割された領域内で、店を探す
	for (int feature_id = 0; feature_id < NUM_FEATURES; ++feature_id) {
		for (int cell_id = 0; cell_id < CITY_SIZE * CITY_SIZE; ++cell_id) {
			int r = cell_id / CITY_SIZE;
			int c = cell_id % CITY_SIZE;

			if (zoningPLan->zones[r][c].type - 1 == feature_id) {
				queue[queue_end++] = Point2DAndFeature(c, r, feature_id);
				distanceMap->distances[r][c][feature_id] = 0;
			}
		}
	}


	// 距離マップを生成
	while (queue_begin < queue_end) {
		Point2DAndFeature pt = queue[queue_begin++];

		int d = distanceMap->distances[pt.y][pt.x][pt.featureId];

		if (pt.y > 0) {
			int old = atomicMin(&distanceMap->distances[pt.y-1][pt.x][pt.featureId], d + 1);
			if (old > d + 1) {
				queue[queue_end++] = Point2DAndFeature(pt.x, pt.y-1, pt.featureId);
			}
		}
		if (pt.y < CITY_SIZE - 1) {
			int old = atomicMin(&distanceMap->distances[pt.y+1][pt.x][pt.featureId], d + 1);
			if (old > d + 1) {
				queue[queue_end++] = Point2DAndFeature(pt.x, pt.y+1, pt.featureId);
			}
		}
		if (pt.x > 0) {
			int old = atomicMin(&distanceMap->distances[pt.y][pt.x-1][pt.featureId], d + 1);
			if (old > d + 1) {
				queue[queue_end++] = Point2DAndFeature(pt.x-1, pt.y, pt.featureId);
			}
		}
		if (pt.x < CITY_SIZE - 1) {
			int old = atomicMin(&distanceMap->distances[pt.y][pt.x+1][pt.featureId], d + 1);
			if (old > d + 1) {
				queue[queue_end++] = Point2DAndFeature(pt.x+1, pt.y, pt.featureId);
			}
		}
	}
}

int main()
{
	time_t start, end;


	ZoningPlan* hostZoningPlan = (ZoningPlan*)malloc(sizeof(ZoningPlan));
	DistanceMap* hostDistanceMap = (DistanceMap*)malloc(sizeof(DistanceMap));
	DistanceMap* hostDistanceMap2 = (DistanceMap*)malloc(sizeof(DistanceMap));

	// 距離を初期化
	memset(hostDistanceMap, 9999, sizeof(DistanceMap));
	memset(hostDistanceMap2, 9999, sizeof(DistanceMap));

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
	
	// デバッグ用
	if (CITY_SIZE <= 8) {
		for (int r = CITY_SIZE - 1; r >= 0; --r) {
			for (int c = 0; c < CITY_SIZE; ++c) {
				printf("%d, ", hostZoningPlan->zones[r][c].type);
			}
			printf("\n");
		}
		printf("\n");
	}

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

	// 距離マップ用に、デバイスバッファを確保
	DistanceMap* devDistanceMap;
	cudaMalloc((void**)&devDistanceMap, sizeof(DistanceMap));


	///////////////////////////////////////////////////////////////////////
	// シングルスレッドで、直近の店までの距離を計算

	// 距離をデバイスバッファへコピー
	cudaMemcpy(devDistanceMap, hostDistanceMap2, sizeof(DistanceMap), cudaMemcpyHostToDevice);

	// スコアの直近の店までの距離を計算
	start = clock();
	computeDistanceToStoreBySingleThread<<<1, 1>>>(devZoningPlan, devDistanceMap);
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
	computeDistanceToStore<<<NUM_GPU_BLOCKS, NUM_GPU_THREADS>>>(devZoningPlan, devDistanceMap);
	end = clock();
	printf("computeDistanceToStore: %lf\n", (double)(end-start)/CLOCKS_PER_SEC);

	// 距離をCPUバッファへコピー
	cudaMemcpy(hostDistanceMap, devDistanceMap, sizeof(DistanceMap), cudaMemcpyDeviceToHost);


	
	// シングルスレッドとマルチスレッドの結果を比較
	for (int r = CITY_SIZE - 1; r >= 0; --r) {
		for (int c = 0; c < CITY_SIZE; ++c) {
			for (int k = 0; k < NUM_FEATURES; ++k) {
				if (hostDistanceMap->distances[r][c][k] != hostDistanceMap2->distances[r][c][k]) {
					printf("ERROR!\n");
				}
			}
		}
	}
	printf("\n");

	// デバッグ用
	if (CITY_SIZE <= 8) {
		for (int r = CITY_SIZE - 1; r >= 0; --r) {
			for (int c = 0; c < CITY_SIZE; ++c) {
				printf("%d, ", hostDistanceMap->distances[r][c][4]);
			}
			printf("\n");
		}
		printf("\n");

		for (int r = CITY_SIZE - 1; r >= 0; --r) {
			for (int c = 0; c < CITY_SIZE; ++c) {
				printf("%d, ", hostDistanceMap2->distances[r][c][4]);
			}
			printf("\n");
		}
		printf("\n");
	}


	// デバイスバッファの開放
	cudaFree(devZoningPlan);
	cudaFree(devDistanceMap);

	// CPUバッファの開放
	free(hostZoningPlan);
	free(hostDistanceMap);
	free(hostDistanceMap2);

	cudaDeviceReset();
}
