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

#define CELL_LENGTH 100
#define CITY_SIZE 20 //200
#define GPU_BLOCK_SIZE 20 //40
#define GPU_NUM_THREADS 1 //96
#define GPU_BLOCK_SCALE (1.0)
#define NUM_FEATURES 5
#define QUEUE_MAX 1999
#define MAX_DIST 99

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

__global__
void generateProposal(ZoningPlan* zoningPlan, unsigned int* randx, int2* cell1, int2* cell2) {
	while (true) {
		int x1 = randf(randx, 0, CITY_SIZE);
		int y1 = randf(randx, 0, CITY_SIZE);
		int x2 = randf(randx, 0, CITY_SIZE);
		int y2 = randf(randx, 0, CITY_SIZE);

		//if (zoningPlan->zones[y1][x1].type != 0 || zoningPlan->zones[y2][x2].type != 0) {
		if (zoningPlan->zones[y1][x1].type == 2 || zoningPlan->zones[y2][x2].type == 2) {
			// swap zone
			int tmp_type = zoningPlan->zones[y1][x1].type;
			zoningPlan->zones[y1][x1].type = zoningPlan->zones[y2][x2].type;
			zoningPlan->zones[y2][x2].type = tmp_type;

			*cell1 = make_int2(x1, y1);
			*cell2 = make_int2(x2, y2);

			break;
		}
	}
}

/**
 * 直近の店までの距離を計算する（マルチスレッド版、shared memory使用）
 */
__global__
void computeDistanceToStore(ZoningPlan* zoningPlan, DistanceMap* distanceMap) {
	__shared__ int sDist[(int)(GPU_BLOCK_SIZE * GPU_BLOCK_SCALE)][(int)(GPU_BLOCK_SIZE * GPU_BLOCK_SCALE)][NUM_FEATURES];
	__shared__ uint3 sQueue[QUEUE_MAX + 1];
	__shared__ unsigned int queue_begin;
	__shared__ unsigned int queue_end;

	queue_begin = 0;
	queue_end = 0;
	__syncthreads();

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
	while ((q_index = atomicInc(&queue_begin, QUEUE_MAX)) < queue_end) {
	//while (queue_begin < queue_end) {
		uint3 pt = sQueue[q_index];
		if (pt.x < 0 || pt.x >= GPU_BLOCK_SIZE * GPU_BLOCK_SCALE || pt.y < 0 || pt.y >= GPU_BLOCK_SIZE * GPU_BLOCK_SCALE) continue;

		int d = sDist[pt.y][pt.x][pt.z];

		if (pt.y > 0) {
			unsigned int old = atomicMin(&sDist[pt.y-1][pt.x][pt.z], d + 1);
			if (old > d + 1) {
				q_index = atomicInc(&queue_end, QUEUE_MAX);
				sQueue[q_index] = make_uint3(pt.x, pt.y-1, pt.z);
			}
		}
		if (pt.y < CITY_SIZE - 1) {
			unsigned int old = atomicMin(&sDist[pt.y+1][pt.x][pt.z], d + 1);
			if (old > d + 1) {
				q_index = atomicInc(&queue_end, QUEUE_MAX);
				sQueue[q_index] = make_uint3(pt.x, pt.y+1, pt.z);
			}
		}
		if (pt.x > 0) {
			unsigned int old = atomicMin(&sDist[pt.y][pt.x-1][pt.z], d + 1);
			if (old > d + 1) {
				q_index = atomicInc(&queue_end, QUEUE_MAX);
				sQueue[q_index] = make_uint3(pt.x-1, pt.y, pt.z);
			}
		}
		if (pt.x < CITY_SIZE - 1) {
			unsigned int old = atomicMin(&sDist[pt.y][pt.x+1][pt.z], d + 1);
			if (old > d + 1) {
				q_index = atomicInc(&queue_end, QUEUE_MAX);
				sQueue[q_index] = make_uint3(pt.x+1, pt.y, pt.z);
			}
		}
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
}

__device__
float min3(int distToStore, int distToAmusement, int distToFactory) {
	return min(min(distToStore, distToAmusement), distToFactory);
}

__global__
void computeScore(ZoningPlan* zoningPlan, DistanceMap* distanceMap, float* devScore, float* devScores) {
	int num_strides = (GPU_BLOCK_SIZE * GPU_BLOCK_SIZE + GPU_NUM_THREADS - 1) / GPU_NUM_THREADS;

	__shared__ float sScore;
	sScore = 0.0f;
	__syncthreads();

	__shared__ float preference[10][9];
	preference[0][0] = 0; preference[0][1] = 0; preference[0][2] = 0; preference[0][3] = 0; preference[0][4] = 0; preference[0][5] = 0; preference[0][6] = 0; preference[0][7] = 1.0; preference[0][8] = 0;
	/*
	preference[0][0] = 0; preference[0][1] = 0; preference[0][2] = 0.15; preference[0][3] = 0.15; preference[0][4] = 0.3; preference[0][5] = 0; preference[0][6] = 0.1; preference[0][7] = 0.1; preference[0][8] = 0.2;
	preference[1][0] = 0; preference[1][1] = 0; preference[1][2] = 0.15; preference[1][3] = 0; preference[1][4] = 0.55; preference[1][5] = 0; preference[1][6] = 0.2; preference[1][7] = 0.1; preference[1][8] = 0;
	preference[2][0] = 0; preference[2][1] = 0; preference[2][2] = 0.05; preference[2][3] = 0; preference[2][4] = 0; preference[2][5] = 0; preference[2][6] = 0.25; preference[2][7] = 0.1; preference[2][8] = 0.6;
	preference[3][0] = 0.18; preference[3][1] = 0.17; preference[3][2] = 0; preference[3][3] = 0.17; preference[3][4] = 0; preference[3][5] = 0.08; preference[3][6] = 0.2; preference[3][7] = 0.2; preference[3][8] = 0;
	preference[4][0] = 0.3; preference[4][1] = 0; preference[4][2] = 0.3; preference[4][3] = 0.1; preference[4][4] = 0; preference[4][5] = 0; preference[4][6] = 0.1; preference[4][7] = 0.2; preference[4][8] = 0;
	preference[5][0] = 0.05; preference[5][1] = 0; preference[5][2] = 0.1; preference[5][3] = 0.2; preference[5][4] = 0.1; preference[5][5] = 0; preference[5][6] = 0.1; preference[5][7] = 0.15; preference[5][8] = 0.3;
	preference[6][0] = 0.15; preference[6][1] = 0.1; preference[6][2] = 0; preference[6][3] = 0.15; preference[6][4] = 0; preference[6][5] = 0.1; preference[6][6] = 0.1; preference[6][7] = 0.2; preference[6][8] = 0.2;
	preference[7][0] = 0.2; preference[7][1] = 0; preference[7][2] = 0.25; preference[7][3] = 0; preference[7][4] = 0.15; preference[7][5] = 0; preference[7][6] = 0.1; preference[7][7] = 0.1; preference[7][8] = 0.2;
	preference[8][0] = 0.3; preference[8][1] = 0; preference[8][2] = 0.15; preference[8][3] = 0.05; preference[8][4] = 0; preference[8][5] = 0; preference[8][6] = 0.25; preference[8][7] = 0.25; preference[8][8] = 0;
	preference[9][0] = 0.4; preference[9][1] = 0; preference[9][2] = 0.2; preference[9][3] = 0; preference[9][4] = 0; preference[9][5] = 0; preference[9][6] = 0.2; preference[9][7] = 0.2; preference[9][8] = 0;
	*/

	//const float ratioPeople[10] = {0.06667, 0.06667, 0.06667, 0.21, 0.09, 0.09, 0.09, 0.12, 0.1, 0.1};
	const float ratioPeople[10] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

	float K[] = {0.002, 0.002, 0.001, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001};

	float lScore = 0.0f;

	int r0 = blockIdx.y * GPU_BLOCK_SIZE;
	int c0 = blockIdx.x * GPU_BLOCK_SIZE;
	for (int i = 0; i < num_strides; ++i) {
		float tmpScore = 0.0f;
		int r1 = (i * GPU_NUM_THREADS + threadIdx.x) / GPU_BLOCK_SIZE;
		int c1 = (i * GPU_NUM_THREADS + threadIdx.x) % GPU_BLOCK_SIZE;

		// 対象ブロックの外ならスキップ
		if (r1 >= GPU_BLOCK_SIZE * GPU_BLOCK_SCALE || c1 >= GPU_BLOCK_SIZE * GPU_BLOCK_SCALE) continue;

		// city範囲の外ならスキップ
		if (r0 + r1 < 0 || r0 + r1 >= CITY_SIZE || c0 + c1 < 0 && c0 + c1 >= CITY_SIZE) continue;

		// 住宅ゾーン以外なら、スキップ
		if (zoningPlan->zones[r0 + r1][c0 + c1].type > 0) continue;

		//for (int peopleType = 0; peopleType < 10; ++peopleType) {
		for (int peopleType = 0; peopleType < 1; ++peopleType) {
			tmpScore += exp(-K[0] * distanceMap->distances[r0 + r1][c0 + c1][0] * CELL_LENGTH) * preference[peopleType][0] * ratioPeople[peopleType]; // 店
			tmpScore += exp(-K[1] * distanceMap->distances[r0 + r1][c0 + c1][4] * CELL_LENGTH) * preference[peopleType][1] * ratioPeople[peopleType]; // 学校
			tmpScore += exp(-K[2] * distanceMap->distances[r0 + r1][c0 + c1][0] * CELL_LENGTH) * preference[peopleType][2] * ratioPeople[peopleType]; // レストラン
			tmpScore += exp(-K[3] * distanceMap->distances[r0 + r1][c0 + c1][2] * CELL_LENGTH) * preference[peopleType][3] * ratioPeople[peopleType]; // 公園
			tmpScore += exp(-K[4] * distanceMap->distances[r0 + r1][c0 + c1][3] * CELL_LENGTH) * preference[peopleType][4] * ratioPeople[peopleType]; // アミューズメント
			tmpScore += exp(-K[5] * distanceMap->distances[r0 + r1][c0 + c1][4] * CELL_LENGTH) * preference[peopleType][5] * ratioPeople[peopleType]; // 図書館
			tmpScore += (1.0f - exp(-K[6] * min3(distanceMap->distances[r0 + r1][c0 + c1][0] * CELL_LENGTH, distanceMap->distances[r0 + r1][c0 + c1][3] * CELL_LENGTH, distanceMap->distances[r0 + r1][c0 + c1][1] * CELL_LENGTH))) * preference[peopleType][6] * ratioPeople[peopleType]; // 騒音
			tmpScore += (1.0f - exp(-K[7] * distanceMap->distances[r0 + r1][c0 + c1][1] * CELL_LENGTH)) * preference[peopleType][7] * ratioPeople[peopleType]; // 汚染
		}
		lScore += tmpScore;

		devScores[(r0 + r1) * CITY_SIZE + c0 + c1] = tmpScore;
	}

	atomicAdd(&sScore, lScore);

	__syncthreads();

	atomicAdd(devScore, sScore);
}

__global__
void acceptProposal(ZoningPlan* zoningPlan, float* score, float* proposalScore, int2* cell1, int2* cell2, int* result) {
	if (*proposalScore > *score) {
		*score = *proposalScore;
		*result = 1;
	} else {
		// プランを元に戻す
		int tmp_type = zoningPlan->zones[cell1->y][cell1->x].type;
		zoningPlan->zones[cell1->y][cell1->x].type = zoningPlan->zones[cell2->y][cell2->x].type;
		zoningPlan->zones[cell2->y][cell2->x].type = tmp_type;
		*result = 0;
	}
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
	zoneTypeDistribution[0] = 0.5f; // 住宅
	zoneTypeDistribution[1] = 0.2f; // 商業
	zoneTypeDistribution[2] = 0.1f; // 工場
	zoneTypeDistribution[3] = 0.1f; // 公園
	zoneTypeDistribution[4] = 0.05f; // アミューズメント
	zoneTypeDistribution[5] = 0.05f; // 学校・図書館
	
	// 初期プランを生成
	start = clock();
	generateZoningPlan(*hostZoningPlan, zoneTypeDistribution);
	end = clock();
	printf("generateZoningPlan: %lf\n", (double)(end-start)/CLOCKS_PER_SEC);
	
	// デバッグ用
	if (CITY_SIZE <= 100) {
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
	CUDA_CALL(cudaMalloc((void**)&devZoningPlan, sizeof(ZoningPlan)));
	CUDA_CALL(cudaMemcpy(devZoningPlan, hostZoningPlan, sizeof(ZoningPlan), cudaMemcpyHostToDevice));

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

	unsigned int* devRand;
	CUDA_CALL(cudaMalloc((void**)&devRand, sizeof(unsigned int)));
	CUDA_CALL(cudaMemset(devRand, 0, sizeof(unsigned int)));

	// 現在プランのスコア
	float* devScore;
	CUDA_CALL(cudaMalloc((void**)&devScore, sizeof(float)));
	CUDA_CALL(cudaMemset(devScore, 0, sizeof(float)));

	// 提案プランのスコア
	float* devProposalScore;
	CUDA_CALL(cudaMalloc((void**)&devProposalScore, sizeof(float)));

	// 交換セル
	int2* devCell1;
	CUDA_CALL(cudaMalloc((void**)&devCell1, sizeof(int2)));
	int2* devCell2;
	CUDA_CALL(cudaMalloc((void**)&devCell2, sizeof(int2)));

	//
	int* devResult;
	CUDA_CALL(cudaMalloc((void**)&devResult, sizeof(int)));



	float* devScores;
	CUDA_CALL(cudaMalloc((void**)&devScores, sizeof(float) * CITY_SIZE * CITY_SIZE));









	printf("start...\n");

	///////////////////////////////////////////////////////////////////////
	// warmp up
	//computeDistanceToStore<<<dim3(CITY_SIZE / GPU_BLOCK_SIZE, CITY_SIZE / GPU_BLOCK_SIZE), GPU_NUM_THREADS>>>(devZoningPlan, devDistanceMap);

	// マルチスレッドで、直近の店までの距離を計算
	start = clock();
	for (int iter = 0; iter < 1000; ++iter) {
		generateProposal<<<1, 1>>>(devZoningPlan, devRand, devCell1, devCell2);

		// 交換したセルを表示
		/*
		int2 cell1, cell2;
		cudaMemcpy(&cell1, devCell1, sizeof(int2), cudaMemcpyDeviceToHost);
		cudaMemcpy(&cell2, devCell2, sizeof(int2), cudaMemcpyDeviceToHost);
		printf("%d,%d <-> %d,%d\n", cell1.x, cell1.y, cell2.x, cell2.y);
		*/

		// 現在のゾーンプランを表示
		/*
		CUDA_CALL(cudaMemcpy(hostZoningPlan, devZoningPlan, sizeof(ZoningPlan), cudaMemcpyDeviceToHost));
		for (int r = CITY_SIZE - 1; r >= 0; --r) {
			for (int c = 0; c < CITY_SIZE; ++c) {
				printf("%d, ", hostZoningPlan->zones[r][c].type);
			}
			printf("\n");
		}
		*/

		computeDistanceToStore<<<dim3(CITY_SIZE / GPU_BLOCK_SIZE, CITY_SIZE / GPU_BLOCK_SIZE), GPU_NUM_THREADS>>>(devZoningPlan, devDistanceMap);
		cudaDeviceSynchronize();

		CUDA_CALL(cudaMemset(devProposalScore, 0, sizeof(float)));
		computeScore<<<dim3(CITY_SIZE / GPU_BLOCK_SIZE, CITY_SIZE / GPU_BLOCK_SIZE), GPU_NUM_THREADS>>>(devZoningPlan, devDistanceMap, devProposalScore, devScores);
		cudaDeviceSynchronize();

		// 提案プランのスコアを表示
		/*
		float proposalScore;
		CUDA_CALL(cudaMemcpy(&proposalScore, devProposalScore, sizeof(float), cudaMemcpyDeviceToHost));
		printf("Score: %lf\n", proposalScore);
		*/

		acceptProposal<<<1, 1>>>(devZoningPlan, devScore, devProposalScore, devCell1, devCell2, devResult);

		// Accept/reject結果を表示
		/*
		int result;
		CUDA_CALL(cudaMemcpy(&result, devResult, sizeof(int), cudaMemcpyDeviceToHost));
		printf("Accept? %d\n", result);
		*/

	}
	end = clock();
	printf("computeDistanceToStore GPU: %lf\n", (double)(end-start)/CLOCKS_PER_SEC);

	// 距離をCPUバッファへコピー
	computeDistanceToStore<<<dim3(CITY_SIZE / GPU_BLOCK_SIZE, CITY_SIZE / GPU_BLOCK_SIZE), GPU_NUM_THREADS>>>(devZoningPlan, devDistanceMap);
cudaDeviceSynchronize();
	CUDA_CALL(cudaMemcpy(hostDistanceMap, devDistanceMap, sizeof(DistanceMap), cudaMemcpyDeviceToHost));

	for (int r = CITY_SIZE - 1; r >= 0; --r) {
		for (int c = 0; c < CITY_SIZE; ++c) {
			printf("%d, ", hostDistanceMap->distances[r][c][1]);
		}
		printf("\n");
	}
	printf("\n");

	// ゾーンプランをCPUバッファへコピー
	CUDA_CALL(cudaMemcpy(hostZoningPlan, devZoningPlan, sizeof(ZoningPlan), cudaMemcpyDeviceToHost));
	for (int r = CITY_SIZE - 1; r >= 0; --r) {
		for (int c = 0; c < CITY_SIZE; ++c) {
			printf("%d, ", hostZoningPlan->zones[r][c].type);
		}
		printf("\n");
	}

	FILE* fp = fopen("zone.txt", "w");
	for (int r = 0; r < CITY_SIZE; ++r) {
		for (int c = 0; c < CITY_SIZE; ++c) {
			fprintf(fp, "%d,", hostZoningPlan->zones[r][c].type);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);



	// ゾーンプランをスコアを表示
	CUDA_CALL(cudaMemset(devProposalScore, 0, sizeof(float)));
	CUDA_CALL(cudaMemset(devScores, 0, sizeof(float) * CITY_SIZE * CITY_SIZE));
	computeScore<<<dim3(CITY_SIZE / GPU_BLOCK_SIZE, CITY_SIZE / GPU_BLOCK_SIZE), GPU_NUM_THREADS>>>(devZoningPlan, devDistanceMap, devProposalScore, devScores);
	cudaDeviceSynchronize();
	float scores[CITY_SIZE * CITY_SIZE];
	float score;
	CUDA_CALL(cudaMemcpy(scores, devScores, sizeof(float) * CITY_SIZE * CITY_SIZE, cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(&score, devProposalScore, sizeof(float), cudaMemcpyDeviceToHost));
	printf("Score: %lf\n", score);
	for (int r = CITY_SIZE - 1; r >= 0; --r) {
		for (int c = 0; c < CITY_SIZE; ++c) {
			printf("%lf, ", scores[r * CITY_SIZE + c]);
		}
		printf("\n");
	}




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
