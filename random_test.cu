#include <stdio.h>
#include <curand.h>
#include <fstream>
#include <iostream>

#define GRID_SIZE	1
#define BLOCK_SIZE	8
#define NUM_TRY		30000

/**
 * 線形乱数生成器
 */
__device__
unsigned int rand(unsigned int* randx) {
    *randx = *randx * 1103515245 + 12345;
    return (*randx)&2147483647;
}

__device__
float randf(unsigned int* randx) {
	return rand(randx) / (float(2147483647) + 1);
}

__device__
float randf(unsigned int* randx, float a, float b) {
	return randf(randx) * (b - a) + a;
}

/**
 * 乱数を生成し、デバイスバッファに格納する
 */
__global__
void gen_random(int* devResults) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	// 乱数シードの初期化
	unsigned int randx = idx;

	for (int i = 0; i < NUM_TRY; ++i) {
		int x = randf(&randx, 0, 200);

		devResults[idx * NUM_TRY + i + 0] = x;
	}
}
      
int main()
{
    int* results;
    int* devResults;
      
	// CPU側でメモリを確保する
    results = new int[GRID_SIZE * BLOCK_SIZE * NUM_TRY];

	// GPU側でメモリを確保する
    cudaMalloc((void**)&devResults, sizeof(int) * GRID_SIZE * BLOCK_SIZE * NUM_TRY);

	// GPU側の関数を呼び出す。（）内が、そのまま関数の引数となる
    gen_random<<<GRID_SIZE, BLOCK_SIZE>>>(devResults);

	// 指定したsize分、GPUのd_bufferから、CPUのbufferへ、データを転送する
    cudaMemcpy(results, devResults, sizeof(int) * GRID_SIZE * BLOCK_SIZE * NUM_TRY, cudaMemcpyDeviceToHost);

	// GPU側で確保したメモリを開放する
    cudaFree(devResults);

	// 結果を表示する
	for (int i = 0; i < GRID_SIZE * BLOCK_SIZE; ++i) {
		char filename[256];
		sprintf(filename, "random_%d.txt", i);
		std::ofstream out(filename);
		for (int j = 0; j < NUM_TRY; ++j) {
			out << results[i * NUM_TRY + j + 0] << std::endl;
			if (results[i * NUM_TRY + j + 0] >= 200) {
				printf("ERROR: x >= 200\n");
			}
			if (results[i * NUM_TRY + j + 1] >= 200) {
				printf("ERROR: y >= 200\n");
			}
		}
		out.close();
		printf("random numbers were written to %s\n", filename);
	}

	// 出力したrandom_X.txtを、plot.pyでグラフにプロットすると、
	// きれいにランダムにプロットが生成されていることが分かるよ！！
	// ヒストグラムを表示して、ほぼ均一に生成されていることが分かる。
	// 結論：Rand()関数は、簡単な乱数生成器として、十分使える！

	// CPU側で確保したメモリを開放する
    free(results);
}
  