
#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define TILEWIDTH 32

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  __shared__ float subTileA[TILEWIDTH][TILEWIDTH];
  __shared__ float subTileB[TILEWIDTH][TILEWIDTH];

  // Store row and column that thread is calculating. 1-1 correspondence from thread to position in tile
  int row = blockIdx.y * TILEWIDTH + threadIdx.y;
  int col = blockIdx.x * TILEWIDTH + threadIdx.x;

  // Running sum
  float runningSum = 0;

  // Loop over each tile in input matrices
  for(int i = 0; i < ceil((float)numAColumns / (float)TILEWIDTH); ++i) {
    // Read the individual tiles into shared memory
    // If thread is within limits of input A matrix
    if(row < numARows && (i * TILEWIDTH + threadIdx.x) < numAColumns) {
      // Read in corresponding element in subtile of input matrices
      subTileA[threadIdx.y][threadIdx.x] = A[row * numAColumns + i * TILEWIDTH + threadIdx.x];
    }
    // Set subtile values to 0 so they do not affect threads inside of output matrix
    else {
      subTileA[threadIdx.y][threadIdx.x] = 0;
    }
    if((i * TILEWIDTH + threadIdx.y) < numBRows && col < numBColumns) {
      subTileB[threadIdx.y][threadIdx.x] = B[(i * TILEWIDTH + threadIdx.y) * numBColumns + col];
    }
    // Set subtile values to 0 so they do not affect threads inside of output matrix
    else {      
      subTileB[threadIdx.y][threadIdx.x] = 0;
    }    

    __syncthreads();  // Make sure all threads are done reading
    
    // Let each thread work on one piece of the subtile
    for(int j = 0; j < TILEWIDTH; ++j) {
      runningSum += subTileA[threadIdx.y][j] * subTileB[j][threadIdx.x];
    }
    
    __syncthreads();  // Make sure all threads are done calculating before moving on
  
  }

  // If thread is calculating something inside output matrix, place runningSum into output matrix
  if(row < numCRows && col < numCColumns) {
    C[row * numCColumns + col] = runningSum;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(numCColumns * numCRows * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  // wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  // wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc(&deviceA, numAColumns * numARows * sizeof(float));
  cudaMalloc(&deviceB, numBColumns * numBRows * sizeof(float));
  cudaMalloc(&deviceC, numCColumns * numCRows * sizeof(float));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, numAColumns * numARows * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBColumns * numBRows * sizeof(float), cudaMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 blocksPerGrid(ceilf((float)numCColumns / (float)TILEWIDTH), ceilf((float)numCRows / (float)TILEWIDTH), 1);
  dim3 threadsPerBlock(TILEWIDTH, TILEWIDTH, 1);

  wbLog(TRACE, "The dimensions of grid is ", blocksPerGrid.x, " x ", blocksPerGrid.y);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<blocksPerGrid, threadsPerBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, numCColumns * numCRows * sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
