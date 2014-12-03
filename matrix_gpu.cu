#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <time.h>

#define MATRIX_SIZE 512 // 行列の１辺の数（1024にすると、俺のマシンだろ落ちちゃう。。。）
#define BLOCK_SIZE 16


__global__
void matrixMul(int* inMatrixA, int* inMatrixB, int* inMatrixC) {
	int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

	/*行列の演算を行う*/
	int total = 0;
	for (int i = 0; i < MATRIX_SIZE; i++) {
		total += inMatrixA[row_idx * MATRIX_SIZE + i] * inMatrixB[i * MATRIX_SIZE + col_idx];

		// オリジナルのコードではシンクロしてるけど、不要だよね。
		//__syncthreads();
	}
	inMatrixC[row_idx * MATRIX_SIZE + col_idx] = total;
}

int main(int argc, char** argv) {
	unsigned int matrixSize = sizeof(unsigned int) * MATRIX_SIZE * MATRIX_SIZE;

	// CPU側のバッファ確保
	int* hMatrixA = (int*)malloc(matrixSize);
	int* hMatrixB = (int*)malloc(matrixSize);
	int* hMatrixC = (int*)malloc(matrixSize);

	/*初期値設定*/
	unsigned int col_idx, row_idx;
	for (col_idx = 0; col_idx < MATRIX_SIZE; col_idx++){
		for (row_idx = 0; row_idx < MATRIX_SIZE; row_idx++){
			hMatrixA[col_idx * MATRIX_SIZE + row_idx] = rand() % 1024;
			hMatrixB[col_idx * MATRIX_SIZE + row_idx] = rand() % 1024;
		}
	}

	// デバイス側のバッファ 
	int* dMatrixA;
	int* dMatrixB;
	int* dMatrixC;
	
	// デバイス側のバッファ確保
	cudaMalloc((void**)&dMatrixA, matrixSize);
	cudaMalloc((void**)&dMatrixB, matrixSize);
	cudaMalloc((void**)&dMatrixC, matrixSize);

	// CPU側バッファからデバイス側バッファへデータ転送
	cudaMemcpy(dMatrixA, hMatrixA, matrixSize, cudaMemcpyHostToDevice);
	cudaMemcpy(dMatrixB, hMatrixB, matrixSize, cudaMemcpyHostToDevice);

	/*ブロックサイズとグリッドサイズの設定*/
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(MATRIX_SIZE/BLOCK_SIZE, MATRIX_SIZE/BLOCK_SIZE);

	// カーネルの起動
	matrixMul<<<grid, block>>>(dMatrixA, dMatrixB, dMatrixC);

	// 結果をデバイス側からCPU側へ転送*/
	cudaMemcpy(hMatrixC, dMatrixC, matrixSize, cudaMemcpyDeviceToHost);

	// CPU側のメモリ開放
	free(hMatrixA);
	free(hMatrixB);
	free(hMatrixC);

	// デバイスメモリ開放
	cudaFree(dMatrixA);
	cudaFree(dMatrixB);
	cudaFree(dMatrixC);

	cudaDeviceReset();
}
