 #include <stdio.h>

__global__
void add(int a, int b, int *c) {
    *c = a+ b;
}

int main(void) {
    int c;
    int *dev_c;

	// GPU側でint型の値を１個格納するためのメモリを確保する
    cudaMalloc((void**)&dev_c, sizeof(int));

	// GPU側の関数を呼び出す。（）内が、そのまま関数の引数となる
    add<<<1,1>>>(12, 2000, dev_c);

	// 指定したsize分、GPUのdev_cから、CPUのcへ、データ（int型の値１個）を転送する
    cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);

	// GPU側で確保したメモリを開放する
    cudaFree(dev_c);

	// 結果を出力する
    printf("12 + 2000 = %d\n", c);

	cudaDeviceReset();

    return 0;
}