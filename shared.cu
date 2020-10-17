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

    extern __shared__ float sh_A_stripe[ ];
    float* sh_B_T_stripe = sh_A_stripe + mid_size * blockDim.x;

    int block_size = blockDim.x * blockDim.y;
    int threadIdxB = blockDim.y * threadIdx.x + threadIdx.y;

    // Copying (to shared memory) the stripe of rows of A, which are neccessary
    // to calculate C[i][j] for current block elements (i, j)
    // To optimize it each thread copies consequtive segment of stripe elements
    int sh_A_stripe_size = blockDim.x * mid_size;
    int elems_per_thread = sh_A_stripe_size % block_size == 0 ?
                            sh_A_stripe_size / block_size :
                           (sh_A_stripe_size + block_size) / block_size;
    int segment_begin = threadIdxB * elems_per_thread;
    int segment_end = segment_begin + elems_per_thread < sh_A_stripe_size ? segment_begin + elems_per_thread : sh_A_stripe_size;
    int stripe_offset = mid_size * ( blockDim.x * blockIdx.x );
    for (int k = segment_begin; k < segment_end; ++k) {
        sh_A_stripe[k] = A[stripe_offset + k];
    }

    // Similar copying for B_T, as previous each thread copies consequtive segment
    int sh_B_T_stripe_size = blockDim.y * mid_size;
    elems_per_thread = sh_B_T_stripe_size % block_size == 0 ?
                        sh_B_T_stripe_size / block_size :
                       (sh_B_T_stripe_size + block_size) / block_size;
    segment_begin = threadIdxB * elems_per_thread;
    segment_end = segment_begin + elems_per_thread < sh_B_T_stripe_size ? segment_begin + elems_per_thread : sh_B_T_stripe_size;
    stripe_offset = mid_size * ( blockDim.y * blockIdx.y );
    for (int k = segment_begin; k < segment_end; ++k) {
        sh_B_T_stripe[k] = B_T[stripe_offset + k];
    }

    __syncthreads();

    // Calculating corresponding to thread element using shared memory
    int threadIdxG_x = blockIdx.x * blockDim.x + threadIdx.x;
    int threadIdxG_y = blockIdx.y * blockDim.y + threadIdx.y;
    int threads_cnt_y = gridDim.y * blockDim.y;
    int tid = threads_cnt_y * threadIdxG_x + threadIdxG_y;
    C[tid] = 0;
    for (int k = 0; k < mid_size; ++k) {
        C[tid] += sh_A_stripe[mid_size * threadIdx.x + k] *
                sh_B_T_stripe[mid_size * threadIdx.y + k];
    }
}

int main() {
    // cond: sizes A and B are matched,
    // height and width are multiples of the block size (block is square),

    const int k_block_side_size = 16;
    const int k_A_height = 128; // to change
    const int k_A_width = 384; // to change
    const int k_B_width = 256; // to change
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

    matrixMul<<< num_blocks_C, block_size, sizeof(float) * k_A_width * 2 * k_block_side_size >>>(d_A, d_B_T, d_C, k_A_width);

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

