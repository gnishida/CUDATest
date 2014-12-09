/**
 * atomicIncのテスト
 */

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <list>
#include <time.h>

#define CITY_SIZE 20
#define GPU_BLOCK_SIZE 20
#define GPU_NUM_THREADS 16
#define NUM_FEATURES 1
#define QUEUE_MAX 1999

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
	__shared__ unsigned int queue_end;

	queue_begin = 0;
	queue_end = 0;
	__syncthreads();

	// global memoryからshared memoryへコピー
	unsigned int q_index = atomicInc(&queue_end, QUEUE_MAX);
	sQueue[q_index] = threadIdx.x;
	__syncthreads();

	while ((q_index = atomicInc(&queue_begin, QUEUE_MAX)) < queue_end) {
		int id = sQueue[q_index];
		results[threadIdx.x] = id;
	}

}

int main()
{
	time_t start, end;

	int* hostResults = (int*)malloc(sizeof(int) * 32);
	int* devResults;
	CUDA_CALL(cudaMalloc((void**)&devResults, sizeof(int) * 32));
	test<<<1, 32>>>(devResults);

	CUDA_CALL(cudaMemcpy(hostResults, devResults, sizeof(int) * 32, cudaMemcpyDeviceToHost));

	for (int i = 0; i < 32; ++i) {
		printf("%d\n", hostResults[i]);
	}
	// デバイスバッファの開放
	cudaFree(devResults);

	// CPUバッファの開放
	free(hostResults);

	cudaDeviceReset();
}
