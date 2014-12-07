#include "ZonePlanMCMC.cuh"
#include <vector>
#include <iostream>

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
float sampleFromCdf(unsigned int* randx, float* cdf, int num) {
	float rnd = randf(randx, 0, cdf[num-1]);

	for (int i = 0; i < num; ++i) {
		if (rnd <= cdf[i]) return i;
	}

	return num - 1;
}

__host__ __device__
float sampleFromPdf(unsigned int* randx, float* pdf, int num) {
	if (num == 0) return 0;

	float cdf[40];
	cdf[0] = pdf[0];
	for (int i = 1; i < num; ++i) {
		if (cdf[i] >= 0) {
			cdf[i] = cdf[i - 1] + pdf[i];
		} else {
			cdf[i] = cdf[i - 1];
		}
	}

	return sampleFromCdf(randx, cdf, num);
}

__host__ __device__
float sum(float* values, int num) {
	float total = 0.0f;
	for (int i = 0; i < num; ++i) {
		total += values[i];
	}
	return total;
}

__host__ __device__
void swapZoneType(zone_type* z1, zone_type* z2) {
	int temp_type = z1->type;
	int temp_level = z1->level;

	z1->type = z2->type;
	z1->level = z2->level;
	z2->type = temp_type;
	z2->level = temp_level;
}

__host__ __device__
float dot2(float* preference, float* feature) {
	float ret = 0.0;
	for (int i = 0; i < 9; ++i) {
		ret += preference[i] * feature[i];
	}
	return ret;
}

__host__ __device__
float noise(float distToFactory, float distToAmusement, float distToStore) {
	float Km = 800.0 - distToFactory;
	float Ka = 400.0 - distToAmusement;
	float Ks = 200.0 - distToStore;

	return max(max(max(Km, Ka), Ks), 0.0f);
}

__host__ __device__
float pollution(float distToFactory) {
	float Km = 800.0 - distToFactory;

	return max(Km, 0.0f);
}

__host__ __device__
void MCMCstep(int numIterations, unsigned int randx, zone_plan* plan, zone_plan* proposal, zone_plan* bestPlan) {
	float K = 0.001;
	
	// preference
	float preference[10][9];
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

	// population ratio
	float ratioPeople[10] = {0.06667, 0.06667, 0.06667, 0.21, 0.09, 0.09, 0.09, 0.12, 0.1, 0.1};

	// poppulation of each level of zone
	float levelPeople[3] = {1, 5, 10};

	// initial plan
	/*
	{
		float zoneTypeDistribution[18] = {0.2, 0.38, 0.2, 0.06, 0.05, 0.03, 0.02, 0.01, 0.01, 0.02, 0, 0, 0.01, 0, 0, 0.01, 0, 0};
		float Z = sum(zoneTypeDistribution, 18);

		float remainedBlockNum[18];
		for (int zi = 0; zi < 18; ++zi) {
			remainedBlockNum[zi] = zoneTypeDistribution[zi] / Z * ZONE_GRID_SIZE * ZONE_GRID_SIZE;
		}

		for (int r = 0; r < ZONE_GRID_SIZE; ++r) {
			for (int c = 0; c < ZONE_GRID_SIZE; ++c) {
				int n = sampleFromPdf(&randx, remainedBlockNum, 18);
				plan->zones[r][c].type = n / 3;
				plan->zones[r][c].level = n % 3 + 1;
				remainedBlockNum[n]--;
			}
		}

		plan->score = 0.0;
	}
	*/

	bestPlan->score = 0.0;
	for (int loop = 0; loop < numIterations; ++loop) {
		// create a proposal
		{
			// copy the current plan to the proposal
			//*proposal = *plan;

			// swap a zone type between two blocks
			/*while (true) {
				int x1 = randf(&randx, 0, ZONE_GRID_SIZE);
				int y1 = randf(&randx, 0, ZONE_GRID_SIZE);
				int x2 = randf(&randx, 0, ZONE_GRID_SIZE);
				int y2 = randf(&randx, 0, ZONE_GRID_SIZE);

				swapZoneType(&proposal->zones[y1][x1], &proposal->zones[y2][x2]);
			}*/
		}

		// 
		proposal->score = 0.0;
		float count = 0.0;

		for (int r = 0; r < ZONE_GRID_SIZE; ++r) {
			for (int c = 0; c < ZONE_GRID_SIZE; ++c) {
				/*
				// skip for non-residential block
				if (plan->zones[r][c].type != 0) continue;

				// compute the distance to the nearest spots
				float distToStore = 4000;
				float distToRestaurant = 4000;
				float distToFactory = 4000;
				float distToPark = 4000;
				float distToAmusement = 4000;
				float distToSchool = 4000;
				float distToLibrary = 4000;

				for (int r2 = 0; r2 < ZONE_GRID_SIZE; ++r2) {
					for (int c2 = 0; c2 < ZONE_GRID_SIZE; ++c2) {
						if (proposal->zones[r2][c2].type == 0) continue;

						//float dist = ZONE_CELL_LEN * sqrtf((r - r2) * (r - r2) + (c - c2) * (c - c2));
						float dist = ZONE_CELL_LEN * (abs(r - r2) + abs(c - c2));
						
						if (proposal->zones[r2][c2].type == 1) { // 店・レストラン
							if (dist < distToStore) {
								distToStore = dist;
								distToRestaurant = dist;
							}
						} else if (proposal->zones[r2][c2].type == 2) { // 工場
							if (dist < distToFactory) {
								distToFactory = dist;
							}
						} else if (proposal->zones[r2][c2].type == 3) { // 公園
							if (dist < distToPark) {
								distToPark = dist;
							}
						} else if (proposal->zones[r2][c2].type == 4) { // アミューズメント
							if (dist < distToAmusement) {
								distToAmusement = dist;
							}
						} else if (proposal->zones[r2][c2].type == 5) { // 学校・図書館
							if (dist < distToSchool) {
								distToSchool = dist;
								distToLibrary = dist;
							}
						}
					}
				}
				
				// compute feature
				float feature[9];
				
				feature[0] = expf(-K * distToStore);
				feature[1] = expf(-K * distToSchool);
				feature[2] = expf(-K * distToRestaurant);
				feature[3] = expf(-K * distToPark);
				feature[4] = expf(-K * distToAmusement);
				feature[5] = expf(-K * distToLibrary);
				feature[6] = expf(-K * noise(distToFactory, distToAmusement, distToStore));
				feature[7] = expf(-K * pollution(distToFactory));
				feature[8] = 0; // 駅はなし
				
				// compute score
				for (int i = 0; i < 10; ++i) {
					proposal->score += dot2(preference[i], feature) * ratioPeople[i] * levelPeople[proposal->zones[r][c].level - 1];
					count += ratioPeople[i] * levelPeople[proposal->zones[r][c].level - 1];
				}
				*/
			}
		}

		//proposal->score /= count;
		
		/*
		// update the best plan
		if (proposal->score > bestPlan->score) {
			*bestPlan = *proposal;
		}

		// compare the current plan and the proposal
		if (proposal->score > plan->score || loop % 10 == 0) {
			// accept
			*plan = *proposal;
		}
		*/
	}
}

/**
 * CUDA version of MCMCM
 */
__global__
void zonePlanMCMCGPUKernel(int* numIterations, zone_plan* plan, zone_plan* proposal, zone_plan* bestPlan) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	// initialize random
	unsigned int randx = idx;

	MCMCstep(*numIterations, randx, &plan[idx], &proposal[idx], &bestPlan[idx]);
}

/**
 * CUDA version of MCMC
 */
__host__
void zonePlanMCMCGPUfunc(zone_plan** bestPlans, int numIterations) {
	// CPU側のメモリを確保
	*bestPlans = (zone_plan*)malloc(sizeof(zone_plan) * ZONE_PLAN_MCMC_GRID_SIZE * ZONE_PLAN_MCMC_BLOCK_SIZE);

	time_t start = clock();

	// デバイスメモリを確保
	int* devNumIterations;
	if (cudaMalloc((void**)&devNumIterations, sizeof(int)) != cudaSuccess) {
		printf("cuda memory allocation error!\n");
		return;
	}
	zone_plan* devPlan;
	if (cudaMalloc((void**)&devPlan, sizeof(zone_plan) * ZONE_PLAN_MCMC_GRID_SIZE * ZONE_PLAN_MCMC_BLOCK_SIZE) != cudaSuccess) {
		printf("cuda memory allocation error!\n");
		return;
	}
	zone_plan* devProposal;
	if (cudaMalloc((void**)&devProposal, sizeof(zone_plan) * ZONE_PLAN_MCMC_GRID_SIZE * ZONE_PLAN_MCMC_BLOCK_SIZE) != cudaSuccess) {
		cudaFree(devPlan);
		printf("cuda memory allocation error!\n");
		return;
	}
	zone_plan* devBestPlan;
	if (cudaMalloc((void**)&devBestPlan, sizeof(zone_plan) * ZONE_PLAN_MCMC_GRID_SIZE * ZONE_PLAN_MCMC_BLOCK_SIZE) != cudaSuccess) {
		cudaFree(devPlan);
		cudaFree(devProposal);
		printf("cuda memory allocation error!\n");
		return;
	}

	time_t end = clock();
	printf("Device memory allocation took %lf\n", (double)(end-start)/CLOCKS_PER_SEC);

	// copy memory
	start = clock();
	if (cudaMemcpy(devNumIterations, &numIterations, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
		cudaFree(devPlan);
		cudaFree(devProposal);
		cudaFree(devBestPlan);
		printf("cuda memory copy error!\n");
		return;
	}
	end = clock();
	printf("Memory copy took %lf\n", (double)(end-start)/CLOCKS_PER_SEC);

	// GPU側のカーネル関数を呼び出す
	start = clock();
    zonePlanMCMCGPUKernel<<<ZONE_PLAN_MCMC_GRID_SIZE, ZONE_PLAN_MCMC_BLOCK_SIZE>>>(devNumIterations, devPlan, devProposal, devBestPlan);
	cudaDeviceSynchronize();
	end = clock();
	printf("Time elapsed: %lf\n", (double)(end-start)/CLOCKS_PER_SEC);

	// 結果をCPU側のバッファへ転送する
	start = clock();
	if (cudaMemcpy(*bestPlans, devBestPlan, sizeof(zone_plan) * ZONE_PLAN_MCMC_GRID_SIZE * ZONE_PLAN_MCMC_BLOCK_SIZE, cudaMemcpyDeviceToHost) != cudaSuccess) {
		cudaFree(devPlan);
		cudaFree(devProposal);
		cudaFree(devBestPlan);
		printf("cuda memory copy error!\n");
		return;
	}
	end = clock();
	printf("Memory copy took %lf\n", (double)(end-start)/CLOCKS_PER_SEC);

	// デバイスメモリを開放する
    cudaFree(devPlan);
    cudaFree(devProposal);
    cudaFree(devBestPlan);

	printf("GPU kernel done.\n");
}

void main() {
	// GPU
	zone_plan* bestPlans;
	zonePlanMCMCGPUfunc(&bestPlans, 10000);

	// CPU
	time_t start = clock();
	zone_plan * plan = (zone_plan*)malloc(sizeof(zone_plan) * ZONE_PLAN_MCMC_GRID_SIZE * ZONE_PLAN_MCMC_BLOCK_SIZE);
	zone_plan * proposal = (zone_plan*)malloc(sizeof(zone_plan) * ZONE_PLAN_MCMC_GRID_SIZE * ZONE_PLAN_MCMC_BLOCK_SIZE);
	zone_plan * bestPlan = (zone_plan*)malloc(sizeof(zone_plan) * ZONE_PLAN_MCMC_GRID_SIZE * ZONE_PLAN_MCMC_BLOCK_SIZE);
	MCMCstep(10000, 0, plan, proposal, bestPlan);
	time_t end = clock();
	printf("CPU version took %lf\n", (double)(end-start)/CLOCKS_PER_SEC);

	cudaDeviceReset();
}