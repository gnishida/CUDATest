#include <stdio.h>
#include <stdlib.h>
#include <curand.h>

#define GRID_SIZE	2
#define BLOCK_SIZE	3

struct point3D {
	float x;
	float y;
	float z;
};

/**
 * GPU側のグローバル関数から呼び出される関数の定義は、deviceを指定する。
 * これで、後は普通の関数定義と同じように、好きな関数を定義できる。
 */
__device__
float negate(float val) {
	return -val;
}

/**
 * GPU側の関数の引数に、構造体を使用できる。
 * これで、コードがスッキリだ！
 */
__global__
void test(point3D* devResults, point3D* devRandom) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	devResults[idx].x = negate(devRandom[idx].x);
	devResults[idx].y = negate(devRandom[idx].y);
	devResults[idx].z = negate(devRandom[idx].z);
}
      
int main()
{
    point3D* results;
    point3D* devResults;
	point3D *devRandom;
      
	// CPU側でメモリを確保する
    results = new point3D[GRID_SIZE * BLOCK_SIZE];

	// CPU側のバッファにデータを格納する
	for (int i = 0; i < GRID_SIZE * BLOCK_SIZE; ++i) {
		results[i].x = (rand() % 100) * 0.01f;
		results[i].y = (rand() % 100) * 0.01f;
		results[i].z = (rand() % 100) * 0.01f;

		printf("%lf, %lf, %lf\n", results[i].x, results[i].y, results[i].z);
	}

	// GPU側でメモリを確保する
	cudaMalloc((void**)&devResults, sizeof(point3D) * GRID_SIZE * BLOCK_SIZE);
    cudaMalloc((void**)&devRandom, sizeof(point3D) * GRID_SIZE * BLOCK_SIZE);

	// CPU側からGPU側へデータを転送する
	cudaMemcpy(devRandom, results, sizeof(point3D) * GRID_SIZE * BLOCK_SIZE, cudaMemcpyHostToDevice);

	// GPU側の関数を呼び出す。（）内が、そのまま関数の引数となる
    test<<<GRID_SIZE, BLOCK_SIZE>>>(devResults, devRandom);

	// 指定したsize分、GPUのd_bufferから、CPUのbufferへ、データを転送する
    cudaMemcpy(results, devResults, sizeof(point3D) * GRID_SIZE * BLOCK_SIZE, cudaMemcpyDeviceToHost);

	// GPU側で確保したメモリを開放する
    cudaFree(devResults);
	cudaFree(devRandom);

	// 結果を表示する
	for (int i = 0; i < GRID_SIZE * BLOCK_SIZE; ++i) {
		printf("%lf, %lf, %lf\n", results[i].x, results[i].y, results[i].z);
	}

	// CPU側で確保したメモリを開放する
    free(results);

	cudaDeviceReset();
}
  