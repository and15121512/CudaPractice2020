#include <iostream>



void fillMatrixA(float* A, int height, int width) {

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            A[width * i + j] = (i == j) ? 1 : 0;
        }
    }

}

void fillMatrixB(float* B, int height, int width) {
    
    fillMatrixA(B, height, width);
}

__global__
void transposeMatrix(float* B, float* B_T, int B_height, int B_width) {
    
    int threadIdxG_x = blockIdx.x * blockDim.x + threadIdx.x;
    int threadIdxG_y = blockIdx.y * blockDim.y + threadIdx.y;
    B_T[B_height * threadIdxG_y + threadIdxG_x] = B[B_width * threadIdxG_x + threadIdxG_y];
}

__global__
void matrixMul(float* A, float* B_T, float* C, const int mid_size) {
    // cond: B must be already transposed,
    //      sizes A, B, C are matched,
    //      mid_size equals to A_width and B_T width,
    //      block must be as square !!!

    int block_side_size = blockDim.x;
    int block_size = blockDim.x * blockDim.x;
    extern __shared__ float sh_A_block[ ];
    float* sh_B_block = sh_A_block + block_size;

    int block_cnt = mid_size / block_side_size;
    for (int matrix_block_idx = 0; matrix_block_idx < block_cnt; ++matrix_block_idx) {
    	int A_global_idx_x = blockIdx.x * block_side_size + threadIdx.x;
	int A_global_idx_y = matrix_block_idx * block_side_size + threadIdx.y;
	int A_global_idx = A_global_idx_x * mid_size + A_global_idx_y;
	sh_A_block[threadIdx.x * block_side_size + threadIdx.y] = A[A_global_idx];

	int B_T_global_idx_x = blockIdx.y * block_side_size + threadIdx.x;
        int B_T_global_idx_y = matrix_block_idx * block_side_size + threadIdx.y;
	int B_T_global_idx = B_T_global_idx_x * mid_size + B_T_global_idx_y;
	sh_B_block[threadIdx.x * block_side_size + threadIdx.y] = B_T[B_T_global_idx];

	__syncthreads();

	int C_global_idx_x = blockIdx.x * block_side_size + threadIdx.x;
        int C_global_idx_y = blockIdx.y * block_side_size + threadIdx.y;
	int C_global_idx = C_global_idx_x * gridDim.y * block_side_size + C_global_idx_y;

	for (int k = 0; k < block_side_size; ++k) {
	    C[C_global_idx] += sh_A_block[threadIdx.x * block_side_size + k] * 
		    		sh_B_block[threadIdx.y * block_side_size + k];
	}	
    }
}

int main() {
    // cond: sizes A and B are matched,
    // height and width are multiples of the block size (block is square),

    const int k_block_side_size = 16;
    const int k_A_height = 256; // to change
    const int k_A_width = 128; // to change
    const int k_B_width = 384; // to change
    const int k_B_height = k_A_width;
    const int k_C_height = k_A_height;
    const int k_C_width = k_B_width;

    float* h_A = new float[k_A_height * k_A_width];
    float* h_B = new float[k_B_height * k_B_width];
    float* h_C = new float[k_C_height * k_C_width];

    fillMatrixA(h_A, k_A_height, k_A_width);
    fillMatrixB(h_B, k_B_height, k_B_width);

    float* d_A;
    float* d_B;
    float* d_B_T;
    float* d_C;

    cudaMalloc(&d_A, sizeof(float) * k_A_height * k_A_width);
    cudaMalloc(&d_B, sizeof(float) * k_B_height * k_B_width);
    cudaMalloc(&d_B_T, sizeof(float) * k_B_width * k_B_height);
    cudaMalloc(&d_C, sizeof(float) * k_C_height * k_C_width);

    cudaMemcpy(d_A, h_A, sizeof(float) * k_A_height * k_A_width, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * k_B_height * k_B_width, cudaMemcpyHostToDevice);

    dim3 block_size(k_block_side_size, k_block_side_size);
    dim3 num_blocks_B(k_B_height / k_block_side_size, k_B_width / k_block_side_size);

    transposeMatrix<<< num_blocks_B, block_size >>>(d_B, d_B_T, k_B_height, k_B_width);

    dim3 num_blocks_C(k_C_height / k_block_side_size, k_C_width / k_block_side_size);

    matrixMul<<< num_blocks_C, block_size, sizeof(float) * 2 * k_block_side_size * k_block_side_size >>>(d_A, d_B_T, d_C, k_A_width);

    cudaMemcpy(h_C, d_C, sizeof(float) * k_C_height * k_C_width, cudaMemcpyDeviceToHost);

    for (int i = 0; i < k_C_height; ++i) {
	for (int j = 0; j < k_C_width; ++j) {
            std::cout << i << ' ' << j << ' ' << h_C[k_C_width * i + j] << std::endl;
	}
    }

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_B_T);
    cudaFree(d_C);

    return 0;
}

