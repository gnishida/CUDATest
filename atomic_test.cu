/**
 * atomicIncのテスト
 */

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <list>
#include <time.h>

#define GPU_NUM_THREADS 4
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
	__shared__ int lock;
	__shared__ int q_tail;
	lock = 0;
	q_tail = 0;
	__syncthreads();


	for (int i = 0; i < 10; ++i) {
		do {

		} while (atomicCAS(&lock, 0, 1));
		int q_index = atomicInc(q_tail, 99999);
		lock = 0;

		results[q_index] = threadIdx.x;
	}
}

int main()
{
	int* hostResults = (int*)malloc(sizeof(int) * GPU_NUM_THREADS * 10);
	int* devResults;
	CUDA_CALL(cudaMalloc((void**)&devResults, sizeof(int) * GPU_NUM_THREADS * 10));
	test<<<1, GPU_NUM_THREADS>>>(devResults);

	CUDA_CALL(cudaMemcpy(hostResults, devResults, sizeof(int) * GPU_NUM_THREADS * 10, cudaMemcpyDeviceToHost));

	for (int i = 0; i < GPU_NUM_THREADS * 10; ++i) {
		printf("%d\n", hostResults[i]);
	}
	
	// デバイスバッファの開放
	cudaFree(devResults);

	// CPUバッファの開放
	free(hostResults);

	cudaDeviceReset();
}
