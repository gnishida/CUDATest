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

__host__
void showPlan(ZoningPlan* zoningPlan) {
	if (CITY_SIZE > 8) return;

	for (int r = CITY_SIZE - 1; r >= 0; --r) {
		for (int c = 0; c < CITY_SIZE; ++c) {
			printf("%d, ", zoningPlan->zones[r][c].type);
		}
		printf("\n");
	}
	printf("\n");
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
 * 与えれた距離マップに基づき、スコアを計算する
 */
__global__
void computeScore(ZoningPlan* zoningPlan, DistanceMap* distanceMap, int* score) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	int stride = ceilf((float)(CITY_SIZE * CITY_SIZE) / NUM_GPU_BLOCKS / NUM_GPU_THREADS);

	int s = 0;
	for (int i = 0; i < stride; ++i) {
		int cell_id = stride * idx + i;
		int r = cell_id / CITY_SIZE;
		int c = cell_id % CITY_SIZE;

		if (zoningPlan->zones[r][c].type > 0) continue;

		// 当該セルのスコアを計算
		for (int feature_id = 0; feature_id < NUM_FEATURES; ++feature_id) {
			s += distanceMap->distances[r][c][feature_id];
		}

	}

	atomicAdd(score, s);
}

/**
 * 提案に基づいて変更されたプランを元に戻す
 */
__global__
void swapCellsInPlan(ZoningPlan* zoningPlan, Point2D* cell1, Point2D* cell2) {
	ZoneType zoneType1 = zoningPlan->zones[cell1->y][cell1->x];
	ZoneType zoneType2 = zoningPlan->zones[cell2->y][cell2->x];
	zoningPlan->zones[cell1->y][cell1->x] = zoneType2;
	zoningPlan->zones[cell2->y][cell2->x] = zoneType1;
}

int main()
{
	time_t start, end;


	ZoningPlan* hostZoningPlan = (ZoningPlan*)malloc(sizeof(ZoningPlan));
	ZoningPlan* hostBestPlan = (ZoningPlan*)malloc(sizeof(ZoningPlan));
	DistanceMap* hostDistanceMap = (DistanceMap*)malloc(sizeof(DistanceMap));
	DistanceMap* hostDistanceMap2 = (DistanceMap*)malloc(sizeof(DistanceMap));
	int* hostScore = (int*)malloc(sizeof(int));
	*hostScore = 9999;

	// デバイスバッファを確保
	ZoningPlan* devZoningPlan;
	ZoningPlan* devProposalPlan;
	DistanceMap* devDistanceMap;
	int* devScore;
	Point2D* devCell1;
	Point2D* devCell2;
	CUDA_CALL(cudaMalloc((void**)&devZoningPlan, sizeof(ZoningPlan)));
	CUDA_CALL(cudaMalloc((void**)&devProposalPlan, sizeof(ZoningPlan)));
	CUDA_CALL(cudaMalloc((void**)&devDistanceMap, sizeof(DistanceMap)));
	CUDA_CALL(cudaMalloc((void**)&devScore, sizeof(int)));
	CUDA_CALL(cudaMalloc((void**)&devCell1, sizeof(Point2D)));
	CUDA_CALL(cudaMalloc((void**)&devCell2, sizeof(Point2D)));


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
	zoneTypeDistribution.resize(NUM_FEATURES + 1);
	
	// 初期プランを生成
	start = clock();
	generateZoningPlan(*hostZoningPlan, zoneTypeDistribution);
	end = clock();
	printf("generateZoningPlan: %lf\n", (double)(end-start)/CLOCKS_PER_SEC);
	
	// デバッグ用
	showPlan(hostZoningPlan);



	// 初期プランをデバイスバッファへコピー
	CUDA_CALL(cudaMemcpy(devZoningPlan, hostZoningPlan, sizeof(ZoningPlan), cudaMemcpyHostToDevice));

	/*
	// 距離をデバイスバッファへコピー
	cudaMemcpy(devDistanceMap, hostDistanceMap, sizeof(DistanceMap), cudaMemcpyHostToDevice);

	// 直近の店までの距離を並列で計算
	start = clock();
	computeDistanceToStore<<<NUM_GPU_BLOCKS, NUM_GPU_THREADS>>>(devZoningPlan, devDistanceMap);
	end = clock();
	printf("computeDistanceToStore: %lf\n", (double)(end-start)/CLOCKS_PER_SEC);

	cudaDeviceSynchronize();

	// スコアを計算
	// ToDo...
	*/
	
	// MCMCスタート
	int bestScore = 9999;
	start = clock();
	for (int iter = 0; iter < 1000; ++iter) {
		// 交換するセルを選択
		Point2D hostCell1(rand() % CITY_SIZE, rand() % CITY_SIZE);
		Point2D hostCell2(rand() % CITY_SIZE, rand() % CITY_SIZE);

		// 交換セルをデバイスバッファへコピー
		CUDA_CALL(cudaMemcpy(devCell1, &hostCell1, sizeof(Point2D), cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemcpy(devCell2, &hostCell2, sizeof(Point2D), cudaMemcpyHostToDevice));

		// 提案プランを作成
		swapCellsInPlan<<<1, 1>>>(devZoningPlan, devCell1, devCell2);
		cudaDeviceSynchronize();

		// デバッグ用
		CUDA_CALL(cudaMemcpy(hostZoningPlan, devZoningPlan, sizeof(ZoningPlan), cudaMemcpyDeviceToHost));
		showPlan(hostZoningPlan);





		// 距離マップを全て9999に初期化
		CUDA_CALL(cudaMemcpy(devDistanceMap, hostDistanceMap, sizeof(DistanceMap), cudaMemcpyHostToDevice));

		// 現在のスコアをバックアップ
		int oldScore = *hostScore;

		// デバイススコアを0に初期化
		*hostScore = 0;
		CUDA_CALL(cudaMemcpy(devScore, hostScore, sizeof(int), cudaMemcpyHostToDevice));

		
		// 提案プランの距離マップを計算する
		computeDistanceToStore<<<NUM_GPU_BLOCKS, NUM_GPU_THREADS>>>(devZoningPlan, devDistanceMap);
		cudaDeviceSynchronize();

		// 提案プランのスコアを計算する
		computeScore<<<NUM_GPU_BLOCKS, NUM_GPU_THREADS>>>(devZoningPlan, devDistanceMap, devScore);
		cudaDeviceSynchronize();

		// スコアをCPUバッファへコピー
		CUDA_CALL(cudaMemcpy(hostScore, devScore, sizeof(int), cudaMemcpyDeviceToHost));

		if (*hostScore < oldScore) {
			// acceptは特にやることなし

			if (*hostScore < bestScore) {
				bestScore = *hostScore;
				//cudaMemcpy(hostBestPlan, devZoningPlan, sizeof(ZoningPlan), cudaMemcpyDeviceToHost);
			}
		} else {
			// current stateを元に戻す
			*hostScore = oldScore;
			swapCellsInPlan<<<1, 1>>>(devZoningPlan, devCell1, devCell2);
			cudaDeviceSynchronize();
		}

		// デバッグ用
		CUDA_CALL(cudaMemcpy(hostZoningPlan, devZoningPlan, sizeof(ZoningPlan), cudaMemcpyDeviceToHost));
		showPlan(hostZoningPlan);

	}
	end = clock();
	printf("MCMC: %lf\n", (double)(end-start)/CLOCKS_PER_SEC);
	
	// ベストプランを表示
	showPlan(hostBestPlan);


	// デバイスバッファの開放
	cudaFree(devZoningPlan);
	cudaFree(devProposalPlan);
	cudaFree(devDistanceMap);
	cudaFree(devScore);
	cudaFree(devCell1);
	cudaFree(devCell2);

	// CPUバッファの開放
	free(hostZoningPlan);
	free(hostBestPlan);
	free(hostDistanceMap);
	free(hostDistanceMap2);
	free(hostScore);

	cudaDeviceReset();
}
