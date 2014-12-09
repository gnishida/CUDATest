#include <stdio.h>

#define BLOCK_SIZE 16
#define L (BLOCK_SIZE * 512)
#define M (BLOCK_SIZE * 512)
#define N (BLOCK_SIZE * 512)

/**
 * l x mの行列Aと、m x nの行列Bを掛けて、l x nの行列Cを作成する。
 * 各スレッドは、行列Cのi行目、j列目の要素を計算する。
 */
__global__ void matmul(float *A, float *B, float *C, int l, int m, int n) {
	int i, j, k;
	float sum;
	i = blockIdx.y * blockDim.y + threadIdx.y;
	j = blockIdx.x * blockDim.x + threadIdx.x;
	sum = 0.0;
	for (k = 0; k < m; k++) {
		sum += A[i * m + k] * B[k * n + j];
	}
	C[i*n+j] = sum;
}

/**
 * l x mの行列Aと、m x nの行列Bを掛けて、l x nの行列Cを作成する。
 * 各スレッドは、行列CのBLOCK_SIZE x blockIdx.y + threadIdx.y行目、
 * BLOCK_SIZE x blockIdx.x + threadIdx.x列目の要素を計算する。
 */
__global__ void Muld(float* A, float* B, int wA, int wB, float* C) {
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// 行列Aのサブ行列の左上のindex
	int aBegin = wA * BLOCK_SIZE * by;

	// 行列Aのサブ行列の一番最後の右上のindex
	int aEnd = aBegin + wA - 1; 

	// 行列Aのサブ行列のステップサイズ
	int aStep = BLOCK_SIZE;

	// Index of the first sub-matrix of B
	// processed by the block
	int bBegin = BLOCK_SIZE * bx;

	// Step size used to iterate through the
	// sub-matrices of B
	int bStep = BLOCK_SIZE * wB;

	// The element of the block sub-matrix
	// that is computed by the thread
	float Csub = 0; 

	// Loop over all the sub-matrices of A and B
	// required to compute the block sub-matrix
	for (int a = aBegin, b = bBegin; a <= aEnd;	a += aStep, b += bStep) {
		// Shared memory for the sub-matrix of A
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		// Shared memory for the sub-matrix of B
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		// グローバルメモリから共有メモリへ、サブ行列をコピーする
		// （各スレッドが１要素ずつコピーする）
		As[ty][tx] = A[a + wA * ty + tx];
		Bs[ty][tx] = B[b + wB * ty + tx];

		__syncthreads();

		// サブ行列を使って、行列Aと行列Bの積を一部だけ計算する
		for (int k = 0; k < BLOCK_SIZE; ++k)
			Csub += As[ty][k] * Bs[k][tx];

		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		__syncthreads();
	}

	// Write the block sub-matrix to global memory;
	// each thread writes one element
	int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
	C[c + wB * ty + tx] = Csub;
}

__host__
void alloc_matrix(float** Ah, float** Ad, int l, int m) {
	*Ah = (float*)malloc(sizeof(float) * l * m);
	cudaMalloc((void**)Ad, sizeof(float) * l * m);
}

__host__
void init_matrix(float* Ah, int l, int m) {
	for (int i = 0; i < l * m; ++i) {
		Ah[i] = rand() % 100;
	}
}

int main(int argc, char *argv[]) {
	float *Ad, *Bd, *Cd;
	float *Ah, *Bh, *Ch;
	time_t start, end;
	
	// prepare matrix A
	alloc_matrix(&Ah, &Ad, L, M);
	init_matrix(Ah, L, M);
	cudaMemcpy(Ad, Ah, sizeof(float) * L * M, cudaMemcpyHostToDevice);
	// do it again for matrix B
	alloc_matrix(&Bh, &Bd, M, N);
	init_matrix(Bh, M, N);
	cudaMemcpy(Bd, Bh, sizeof(float) * M * N, cudaMemcpyHostToDevice);
	// allocate spaces for matrix C
	alloc_matrix(&Ch, &Cd, L, N);


	// launch matmul kernel
	start = clock();
	matmul<<<dim3(N / BLOCK_SIZE, L / BLOCK_SIZE), dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(Ad, Bd, Cd, L, M, N);
	end = clock();
	printf("Time: %lf\n", (double)(end - start)/CLOCKS_PER_SEC);

	start = clock();
	Muld<<<dim3(N / BLOCK_SIZE, L / BLOCK_SIZE), dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(Ad, Bd, M, N, Cd);
	end = clock();
	printf("Time: %lf\n", (double)(end - start)/CLOCKS_PER_SEC);

	return 0;
}