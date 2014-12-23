/**
 * atomicIncのテスト
 */

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <list>
#include <time.h>

#define GPU_NUM_THREADS 16
#define QUEUE_MAX 39

#define CUDA_CALL(x) {if((x) != cudaSuccess){ \
  printf("CUDA error at %s:%d\n",__FILE__,__LINE__); \
  printf("  %s\n", cudaGetErrorString(cudaGetLastError())); \
  exit(EXIT_FAILURE);}} 






/**
 * 直近の店までの距離を計算する（マルチスレッド版、shared memory使用）
 */
__global__
void test(int* results) {
	__shared__ int sQueue[QUEUE_MAX + 1];
	__shared__ unsigned int queue_begin;

	for (int i = threadIdx.x; i < QUEUE_MAX + 1; i += GPU_NUM_THREADS) {
		sQueue[i] = 99;
	}

	queue_begin = 35;
	__syncthreads();

	for (int i = 0; i < 1; ++i) {
		unsigned int q_index = atomicInc(&queue_begin, QUEUE_MAX);
		sQueue[q_index] = threadIdx.x;
	}
	
	for (int i = threadIdx.x; i < QUEUE_MAX + 1; i += GPU_NUM_THREADS) {
		results[i] = sQueue[i];
	}

}

int main()
{
	int* hostResults = (int*)malloc(sizeof(int) * (QUEUE_MAX + 1));
	int* devResults;
	CUDA_CALL(cudaMalloc((void**)&devResults, sizeof(int) * (QUEUE_MAX + 1)));
	test<<<1, GPU_NUM_THREADS>>>(devResults);

	CUDA_CALL(cudaMemcpy(hostResults, devResults, sizeof(int) * (QUEUE_MAX + 1), cudaMemcpyDeviceToHost));

	for (int i = 0; i < QUEUE_MAX + 1; ++i) {
		printf("%2d: %2d\n", i, hostResults[i]);
	}
	// デバイスバッファの開放
	cudaFree(devResults);

	// CPUバッファの開放
	free(hostResults);

	cudaDeviceReset();
}
