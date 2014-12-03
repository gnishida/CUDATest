#include <stdio.h>
#include <curand.h>

#define GRID_SIZE	32
#define BLOCK_SIZE	512
#define NUM_TRY		10000

/**
 * 乱数に基づいて(x,y)を生成し、円の中に入る確率を計算し、devResultsに格納する。
 */
__global__
void compute_pi(float* devResults, float* devRandom){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int step = gridDim.x * blockDim.x * 2;

	int count = 0;
	for (int iter = 0; iter < NUM_TRY; ++iter) {
		// 乱数に基づいて(x,y)を生成
		float x = devRandom[iter * step + idx * 2];
		float y = devRandom[iter * step + idx * 2 + 1];

		// 円の中に入っているかチェック
		if (x * x + y * y <= 1) {
			count++;
		}
	}

	devResults[idx] = (float)count / NUM_TRY;
}
      
int main()
{
    float* results;
    float* devResults;
	curandGenerator_t gen;
	float *devRandom;
      
	// CPU側でメモリを確保する
    results = new float[GRID_SIZE * BLOCK_SIZE];

	// GPU側でメモリを確保する
    cudaMalloc((void**)&devResults, sizeof(float) * GRID_SIZE * BLOCK_SIZE);
	cudaMalloc((void**)&devRandom, sizeof(float) * GRID_SIZE * BLOCK_SIZE * NUM_TRY * 2);

	// 乱数生成器を作成
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

	// 乱数を生成し、デバイス側のバッファに格納する
	curandGenerateUniform(gen, devRandom, GRID_SIZE * BLOCK_SIZE * NUM_TRY * 2);

	// GPU側の関数を呼び出す。（）内が、そのまま関数の引数となる
    compute_pi<<<GRID_SIZE, BLOCK_SIZE>>>(devResults, devRandom);

	// 指定したsize分、GPUのd_bufferから、CPUのbufferへ、データを転送する
    cudaMemcpy(results, devResults, sizeof(float) * GRID_SIZE * BLOCK_SIZE, cudaMemcpyDeviceToHost);

	// GPU側で確保したメモリを開放する
    cudaFree(devResults);
	cudaFree(devRandom);

	// 結果を表示する
	float count = 0.0;
	for (int i = 0; i < GRID_SIZE * BLOCK_SIZE; ++i) {
		count += results[i];
	}
	printf("PI: %lf\n", count * 4.0 / GRID_SIZE / BLOCK_SIZE);

	// CPU側で確保したメモリを開放する
    free(results);

	cudaDeviceReset();
}
  