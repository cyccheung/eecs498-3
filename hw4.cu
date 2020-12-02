#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define MASK_WIDTH 3
#define MASK_RADIUS 1
#define TILE_SIZE 8
#define BLOCK_SIZE (TILE_SIZE + MASK_WIDTH - 1)
//@@ Define constant memory for device kernel here
__device__ __constant__ float MASK[MASK_WIDTH * MASK_WIDTH * MASK_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  // Calculate which row, column, and depth this thread will be outputting to
  int col_o = blockIdx.x * TILE_SIZE + tx;
  int row_o = blockIdx.y * TILE_SIZE + ty;
  int depth_o = blockIdx.z * TILE_SIZE + tz;
  // Calculate which row, column, and depth this thread will be bringing in
  int col_i = col_o - MASK_RADIUS;
  int row_i = row_o - MASK_RADIUS;
  int depth_i = depth_o - MASK_RADIUS;

  // Shared memory
  __shared__ float N_ds[BLOCK_SIZE][BLOCK_SIZE][BLOCK_SIZE];

  // Load up shared memory, ghost elements = 0
  if( (row_i >= 0) && (row_i < y_size) &&
      (col_i >= 0) && (col_i < x_size) &&
      (depth_i >= 0) && (depth_i < z_size)) {
        N_ds[tz][ty][tx] = input[depth_i * (y_size * x_size) + row_i * x_size + col_i];
  }
  else {
    N_ds[tz][ty][tx] = 0.0f;
  }

  __syncthreads();
  float result = 0.0f;
  if(tz < TILE_SIZE && ty < TILE_SIZE && tx < TILE_SIZE) {
    for(int z_mask = 0; z_mask < MASK_WIDTH; ++z_mask) {
      for(int y_mask = 0; y_mask < MASK_WIDTH; ++y_mask) {
        for(int x_mask = 0; x_mask < MASK_WIDTH; ++x_mask) {
          result += (MASK[z_mask * MASK_WIDTH * MASK_WIDTH + y_mask * MASK_WIDTH + x_mask] * N_ds[tz + z_mask][ty + y_mask][tx + x_mask]);
        }
      }
    }

    if(row_o < y_size && col_o < x_size && depth_o < z_size) {
      output[depth_o * y_size * x_size + row_o * x_size + col_o] = result;
    }
  }
  
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));
  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc(&deviceInput, (z_size + MASK_WIDTH - 1) * (y_size + MASK_WIDTH - 1) * (x_size + MASK_WIDTH - 1) * sizeof(float));
  cudaMalloc(&deviceOutput, z_size * y_size * x_size * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, &hostInput[3], z_size * y_size * x_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(MASK, hostKernel, kernelLength * sizeof(float));
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 blocksPerGrid(ceilf((float)x_size / (float)TILE_SIZE), ceilf((float)y_size / (float)TILE_SIZE), ceilf((float)z_size / (float)TILE_SIZE));
  dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
  wbLog(TRACE, "The dimensions of grid is ", blocksPerGrid.x, " x ", blocksPerGrid.y, " x ", blocksPerGrid.z);
  wbLog(TRACE, "The dimensions of block is ", threadsPerBlock.x, " x ", threadsPerBlock.y, " x ", threadsPerBlock.z);
  //@@ Launch the GPU kernel here
  conv3d<<<blocksPerGrid, threadsPerBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(&hostOutput[3], deviceOutput, z_size * y_size * x_size * sizeof(float), cudaMemcpyDeviceToHost);

  // cudaMemcpy(testMask, MASK, kernelLength * sizeof(float));
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}


