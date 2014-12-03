#include <stdio.h>

#define SIZE_TEXT (sizeof(text)-1)
#define SIZE_END (sizeof(end)-1)

__device__ char text[] =
"__ bottles of beer on the wall, __ bottles of beer!\n"
"Take one down, and pass it around, ## bottles of beer on the wall!\n\n";
       
__device__ char end[] =
"01 bottle of beer on the wall, 01 bottle of beer.\n"
"Take one down and pass it around, no more bottles of beer on the wall.\n"
"\n"
"No more bottles of beer on the wall, no more bottles of beer.\n"
"Go to the store and buy some more, 99 bottles of beer on the wall.";

__global__
void bottle99(char *addr){
    int x = threadIdx.x;
	addr += x * SIZE_TEXT;

    int bottle = 99 - x;
	if (bottle == 1) {
		for (int i=0; i<SIZE_END; i++) {
			addr[i] = end[i];
		}
		addr[SIZE_END] = '\0';
	} else {
		char c1 = (bottle/10) + '0';
		char c2 = (bottle%10) + '0';

		char d1 = ((bottle-1)/10) + '0';
		char d2 = ((bottle-1)%10) + '0';

		// replace '__' by the current bottle number, and replace '##' by the next bottle number
		for (int i=0; i<SIZE_TEXT; i++) {
			int c = text[i];
			if (c == '_') {
				addr[i] = c1;
				addr[i+1] = c2;
				i++;
			} else if (c == '#') {
				addr[i] = d1;
				addr[i+1] = d2;
				i++;
			} else {
				addr[i] = text[i];
			}
        }
    }
}
      
int main()
{
    char *buffer;
    char *d_buffer;
      
    int size = SIZE_TEXT * 98 + SIZE_END + 1;

	// CPU側でメモリを確保する
    buffer = new char[size];

	// GPU側でメモリを確保する
    cudaMalloc((void**)&d_buffer, size);

	// GPU側の関数を呼び出す。（）内が、そのまま関数の引数となる
    bottle99<<<1, 99>>>(d_buffer);

	// 指定したsize分、GPUのd_bufferから、CPUのbufferへ、データを転送する
    cudaMemcpy(buffer, d_buffer, size, cudaMemcpyDeviceToHost);

	// GPU側で確保したメモリを開放する
    cudaFree(d_buffer);

	// 結果を表示する
    puts(buffer);

	// CPU側で確保したメモリを開放する
    free(buffer);

	cudaDeviceReset();
}
  