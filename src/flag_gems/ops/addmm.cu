const int BLOCK_SIZE = 16;

template <typename MATRIX_T>
void __global__ mm_kernel(MATRIX_T matrix_a, MATRIX_T matrix_b, MATRIX_T &matrix_c, int M, int K, int N)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    __shared__ float sh_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sh_b[BLOCK_SIZE][BLOCK_SIZE];

    int xidx = threadIdx.x;
    int yidx = threadIdx.y;
    int tiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float acc = 0.0;
    for (int tile = 0; tile < tiles; tile++)
    {
        sh_a[xidx][yidx] = matrix_a[x * K + tile * BLOCK_SIZE + yidx];
        sh_b[xidx][yidx] = matrix_b[(tile * BLOCK_SIZE + xidx) * N + y]

            __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; k++)
        {
            acc += sh_a[xidx][k] * sh_b[k][yidx];
        }
        __syncthreads();
    }

    matrix_c[x * N + y] = acc;
}