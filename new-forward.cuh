#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#define MAX_MAPS 24
#define MAX_CHANNELS 12
#define MASK_WIDTH 7
#define MASK_RADIUS 3
#define TILE_SIZE 16    // Possibly 26
#define BLOCK_SIZE TILE_SIZE+MASK_WIDTH-1
#define NUMTHREADS BLOCK_SIZE*BLOCK_SIZE
__device__ __constant__ float MASKS[MAX_MAPS * MAX_CHANNELS * MASK_WIDTH * MASK_WIDTH];

namespace mxnet
{
namespace op
{

__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    /*
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */


    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a
    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define MASKS4d(i3, i2, i1, i0) MASKS[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define x_ds4d(i3, i2, i1, i0) x_ds[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]

    // ---------------------------------------------------------

    // My stuff -------------------------
    // Optimization, read in input x into shared memory
    int imageIndex = blockIdx.x; // Image index
    int mapIndex = blockIdx.y; // Output feature map index
    int blockIndex = blockIdx.z;    // Block index in map
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // Calculate which row and column this thread will output to
    int W_grid = ceilf((float)W / (float)TILE_SIZE);
    // H_grid = ceil((float)H / TILE_SIZE);
    int col_o = (blockIndex % W_grid) * TILE_SIZE + tx;
    int row_o = (blockIndex / W_grid) * TILE_SIZE + ty;
    // Calculate which row and column this thread will be bringing in
    int col_i = col_o - MASK_RADIUS;
    int row_i = row_o - MASK_RADIUS;
    // Shared memory
    __shared__ float x_ds[BLOCK_SIZE][BLOCK_SIZE];
    const int H_out = H - K + 1;    // Needed for y4d
    const int W_out = W - K + 1;    // Needed for y4d

    for(int channelIndex = 0; channelIndex < C; ++channelIndex) {
        if( (row_i >= 0) && (row_i < H) && (col_i >= 0) && (col_i < W) ) {
            x_ds[ty][tx] = x4d(imageIndex, channelIndex, row_i, col_i);
        }
        else {
            x_ds[ty][tx] = 0.0f;

        }
        __syncthreads();

        // Shared memory should be filled up by this point

        float result = 0.0f;
        if(ty < TILE_SIZE && tx < TILE_SIZE) {
            for(int y_mask = 0; y_mask < K; ++y_mask) {
                for(int x_mask = 0; x_mask < K; ++x_mask) {
                    result += x_ds[ty + y_mask][tx + x_mask] * MASKS4d(mapIndex, channelIndex, y_mask, x_mask);
                }
            }

            if(row_o < H && col_o < W) {
                y4d(blockIndex, mapIndex, row_o, col_o) = result;
            }
        }
        __syncthreads();
    }

/*
    int n, m, h0, w0, h_base, w_base, h, w, W_grid, H_grid;
    int X_tile_width = TILE_SIZE + K - 1;
    __shared__ float x_ds[BLOCK_SIZE*BLOCK_SIZE];
    W_grid = ceil((float)W / TILE_SIZE);
    H_grid = ceil((float)H / TILE_SIZE);
    n = blockIdx.x;
    m = blockIdx.y;
    h0 = threadIdx.x;
    w0 = threadIdx.y;
    h_base = (blockIdx.z/W_grid) * TILE_SIZE;
    w_base = (blockIdx.z % W_grid) * TILE_SIZE;
    h = h_base + h0;
    w = w_base + w0;

    int c, i, j, p, q;
    for(c = 0; c < C; c++) {
        for(i = h; i < h_base + X_tile_width; i += TILE_SIZE) {
            for(j = w; j < w_base + X_tile_width; j += TILE_SIZE) {
                if( (i - h_base) >= 0 && (i - h_base) < H && 
                    (j - w_base) >= 0 && (j - w_base) < W) {
                    x_ds[(i - h_base) * X_tile_width + j - w_base] = x4d(n, c, h, w);
                }
                else {
                    x_ds[(i - h_base) * X_tile_width + j - w_base] = 0.0f;
                }
            }
        }
        __syncthreads();
        float result = 0.0;
        if(h0 < TILE_SIZE && w0 < TILE_SIZE) {
            for(p = 0; p < K; ++p) {
                for(q = 0; q < K; q++) {
                    result += x_ds[(h + p) * X_tile_width + w + q] * MASKS4d(m, c, p, q);
                }
            }
            __syncthreads();
            const int H_out = H - K + 1;
            const int W_out = W - K + 1;
            y4d(n, m, h, w) = result;
        }
    }
*/
    // My stuff ^^^^^^^^^^^^^^^^^^^^^^^^^
    
    /*
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    if (b < B) // for each image in the batch
    {
        for (int m = 0; m < M; m++)         // for each output feature maps
            for (int h = 0; h < H_out; h++) // for each output element
                for (int w = 0; w < W_out; w++)
                {
                    y4d(b, m, h, w) = 0;
                    float result = 0.0;
                    for (int c = 0; c < C; c++)     // sum over all input feature maps
                        for (int p = 0; p < K; p++) // KxK filter
                            for (int q = 0; q < K; q++)
                                // y4d(b, m, h, w) += x4d(b, c, h + p, w + q) * k4d(m, c, p, q);
                                y4d(b, m, h, w) += x_ds4d(b, c, h + p, w + q) * MASKS4d(m, c, p, q);
                }
    }
    */

#undef y4d
#undef x4d
#undef k4d
}

/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   We only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // // Use mxnet's CHECK_EQ to do assertions.
    // CHECK_EQ(0, 1) 

    const int B = x.shape_[0];
    const int M = y.shape_[1]; // num_filter
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];
    // const int H_out = H - K + 1;
    // const int W_out = W - K + 1;
    int W_grid = ceil((float)W / (float)TILE_SIZE); // Number of blocks wide
    int H_grid = ceil((float)H / (float)TILE_SIZE); // Number of blocks high

    // Copy to MASK constant memory
    cudaMemcpyToSymbol(MASKS, w.dptr_, M*C*K*K * sizeof(float));
    dim3 gridDim(B, M, W_grid * H_grid);    // Num images, num output feature maps per image, num blocks per map
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    forward_kernel<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_, B, M, C, H, W, K);
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
}

/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    assert(0 && "No forward implementation for other datatypes needed");
}
}
}

#endif
