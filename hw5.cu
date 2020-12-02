// Given a list (lst) of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void total(float *input, float *output, int len) {
  // Compute starting index that thread will be loading
  int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  //@@ Load a segment of the input vector into shared memory
  volatile __shared__ float sdata[BLOCK_SIZE];
  if(i < len) {
    if(i + blockDim.x < len) {
      sdata[threadIdx.x] = input[i] + input[i + blockDim.x];
    }
    else {
      sdata[threadIdx.x] = input[i];
    }
  }
  __syncthreads();
  //@@ Traverse the reduction tree
  if(BLOCK_SIZE >= 1024) {
    if(threadIdx.x < 512) {
      sdata[threadIdx.x] += sdata[threadIdx.x + 512];
    }
    __syncthreads();
  }
  if(BLOCK_SIZE >= 512) {
    if(threadIdx.x < 256) {
      sdata[threadIdx.x] += sdata[threadIdx.x + 256];
    }
    __syncthreads();
  }
  if(BLOCK_SIZE >= 256) {
    if(threadIdx.x < 128) {
      sdata[threadIdx.x] += sdata[threadIdx.x + 128];
    }
    __syncthreads();
  }
  if(BLOCK_SIZE >= 128) {
    if(threadIdx.x < 64) {
      sdata[threadIdx.x] += sdata[threadIdx.x + 64];
    }
    __syncthreads();
  }
  if(threadIdx.x < 32) {
    if(BLOCK_SIZE >= 64) {
      sdata[threadIdx.x] += sdata[threadIdx.x + 32];
    }
    if(BLOCK_SIZE >= 32) {
      sdata[threadIdx.x] += sdata[threadIdx.x + 16];
    }
    if(BLOCK_SIZE >= 16) {
      sdata[threadIdx.x] += sdata[threadIdx.x + 8];
    }
    if(BLOCK_SIZE >= 8) {
      sdata[threadIdx.x] += sdata[threadIdx.x + 4];
    }
    if(BLOCK_SIZE >= 3) {
      sdata[threadIdx.x] += sdata[threadIdx.x + 2];
    }
    if(BLOCK_SIZE >= 2) {
      sdata[threadIdx.x] += sdata[threadIdx.x + 1];
    }
  }
  //@@ Write the computed sum of the block to the output vector at the
  //@@ correct index
  output[blockIdx.x] = sdata[0];
}

int main(int argc, char **argv) {
  int ii;
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numInputElements;  // number of elements in the input list
  int numOutputElements; // number of elements in the output list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput =
      (float *)wbImport(wbArg_getInputFile(args, 0), &numInputElements);

  numOutputElements = numInputElements / (BLOCK_SIZE << 1);
  if (numInputElements % (BLOCK_SIZE << 1)) {
    numOutputElements++;
  }
  hostOutput = (float *)malloc(numOutputElements * sizeof(float));

  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numInputElements);
  wbLog(TRACE, "The number of output elements in the input is ",
        numOutputElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc(&deviceInput, numInputElements * sizeof(float));
  cudaMalloc(&deviceOutput, numOutputElements * sizeof(float));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, numInputElements * sizeof(float), cudaMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input memory to the GPU.");
  //@@ Initialize the grid and block dimensions here
  dim3 blocksPerGrid(ceil((float)numInputElements / (2.0*(float)BLOCK_SIZE)), 1, 1);
  dim3 threadsPerBlock(BLOCK_SIZE, 1, 1);
  wbLog(TRACE, "The dimensions of grid is ", blocksPerGrid.x, " x ", blocksPerGrid.y, " x ", blocksPerGrid.z);    // Debugging
  wbLog(TRACE, "The dimensions of block is ", threadsPerBlock.x, " x ", threadsPerBlock.y, " x ", threadsPerBlock.z);   // Debugging

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  total<<<blocksPerGrid, threadsPerBlock>>>(deviceInput, deviceOutput, numInputElements);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput, deviceOutput, numOutputElements * sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  /********************************************************************
   * Reduce output vector on the host
   * NOTE: One could also perform the reduction of the output vector
   * recursively and support any size input. For simplicity, we do not
   * require that for this lab.
   ********************************************************************/
  for (ii = 1; ii < numOutputElements; ii++) {
    hostOutput[0] += hostOutput[ii];
  }

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, 1);

  free(hostInput);
  free(hostOutput);

  return 0;
}
