const int ROW_TILE_WIDTH = 16;
const int COL_TILE_WIDTH = 16;
template <typename MATRIX_T, typename BIAS_T>
__global__ void addmm_kernel(const MATRIX_T *A, const MATRIX_T *B, const BIAS_T *bias, MATRIX_T *C,
                             int M, int N, int K, int alpha, int beta)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // add the shared memory
    __shared__ T shATile[ROW_TILE_WIDTH][COL_TILE_WIDTH];
    __shared__ T shBTile[ROW_TILE_WIDTH][COL_TILE_WIDTH];

    for (int p = 0; p < K / COL_TILE_WIDTH; p += COL_TILE_WIDTH)
    {
        if (threadIdx.y < ROW_TILE_WIDTH && threadIdx.x < COL_TILE_WIDTH)
        {
            shATile[threadIdx.x][threadIdx.y] = A[i * M + p * ROW_TILE_WIDTH + threadIdx.x];
            shBTile[threadIdx.x][threadIdx.y] = B[(p * COL_TILE_WIDTH + threadIdx.y) * N + j];
        }
        __syncthreads();
        if (i < M && j < N)
        {
            T value = 0;
            for (int q = 0; q < COL_TILE_WIDTH; q++)
            {
                value += shATile[threadIdx.x][q] * shBTile[q][threadIdx.y];
            }
        }
        __syncthreads();
    }
    C[i * M + j] = value * alpha + bias[i * M + j] * beta;
}

// if (i < M && j < N)
// {
//     T value = 0;
//     for (int k = 0; k < K; k++)
//     {
//         value += A[i * M + k] * B[k * N + j];
//     }
//     C[i * M + j] = value * alpha + bias[i * M + j] * beta;
// }
