#include <stdio.h>

__device__
unsigned int Rand(unsigned int randx)
{
    randx = randx*1103515245+12345;
    return randx&2147483647;
}

__global__
void generate_plan(int* results){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int randx = idx;

	// ゾーンプランを生成
	/*
	__shared__ int zones[2][2];
	for (int r = 0; r < 2; ++r) {
		for (int c = 0; c < 2; ++c) {
			randx = Rand(randx);
			zones[r][c] = randx % 4; // ゾーンタイプ 0 - 3
		}
	}

	int score = 0;
	for (int r = 0; r < 2; ++r) {
		for (int c = 0; c < 2; ++c) {
			if 
			(c == 0) {
				score += abs(zones[r][c+1] - zones[r][c]);
			} else {
				score += abs(zones[r][c-1] - zones[r][c]);
			}

			if (r == 0) {
				score += abs(zones[r+1][c] - zones[r][c]);
			} else {
				score += abs(zones[r-1][c] - zones[r][c]);
			}
		}
	}*/

	results[idx] = Rand(randx) % 4;
	//results[idx] = score;
}
      
int main()
{
    int *results;
    int *dResults;
      
	// CPU側でメモリを確保する
    results = new int[32];

	// GPU側でメモリを確保する
    cudaMalloc((void**)&dResults, sizeof(int) * 32);

	// GPU側の関数を呼び出す。（）内が、そのまま関数の引数となる
    generate_plan<<<32, 1>>>(dResults);

	// 指定したsize分、GPUのd_bufferから、CPUのbufferへ、データを転送する
    cudaMemcpy(results, dResults, sizeof(int) * 32, cudaMemcpyDeviceToHost);

	// GPU側で確保したメモリを開放する
    cudaFree(dResults);

	// 結果を表示する
	for (int i = 0; i < 32; ++i) {
		for (int j = 0; j < 1; ++j) {
			printf("%d,%d: %d\n", i, j, results[i]);
		}
	}

	// CPU側で確保したメモリを開放する
    free(results);

	cudaDeviceReset();
}
  