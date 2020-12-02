// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256

//@@ insert code here
#define BLOCK_SIZE HISTOGRAM_LENGTH

// Cast the image from float to unsigned char
__global__ void castuc(float *input, unsigned char *output, int width, int height, int channels) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  if(tx < width * height * channels) {
    output[tx] = (unsigned char)(255 * input[tx]);
  }
}

// Sequential castuc
// __global__ void castuc(float *input, unsigned char *output, int width, int height, int channels) {
//   for(int i = 0; i < width * height * channels; ++i) {
//     output[i] = (unsigned char)(255 * input[i]);
//   }
// }

// Convert the image from RGB to Grayscale
// __global__ void convert(unsigned char *input, unsigned char *output, int width, int height, int channels) {
//   int col = threadIdx.x + blockIdx.x * blockDim.x;
//   int row = threadIdx.y + blockIdx.y * blockDim.y;

//   if(col < width && row < height) {
//     int greyOffset = row * width + col;
//     int rgbOffset = greyOffset * channels;
//     unsigned char r = input[rgbOffset];
//     unsigned char g = input[rgbOffset+1];
//     unsigned char b = input[rgbOffset+2];

//     output[greyOffset] = (unsigned char)(0.21f*r + 0.71f*g + 0.07f*b);
//   }
// }

// Sequential convert
__global__ void convert(unsigned char *input, unsigned char *output, int width, int height, int channels) {
  for(int i = 0; i < height; ++i) {
    for(int j = 0; j < width; ++j) {
      int idx = i * width + j;
      unsigned char r = input[channels*idx];
      unsigned char g = input[channels*idx+1];
      unsigned char b = input[channels*idx+2];
      output[idx] = (unsigned char)(0.21f*r + 0.71f*g + 0.07f*b);
    }
  }
}

// Compute histogram of grayImage
__global__ void histogram(unsigned char *input, uint *output, int len) {
  const int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
  const int numThreads = blockDim.x * gridDim.x;
  __shared__ uint partialHistogram[HISTOGRAM_LENGTH];
  // Clear out partial histogram buffer
  for(int pos = threadIdx.x; pos < HISTOGRAM_LENGTH; pos += blockDim.x) {
    partialHistogram[pos] = 0;
  }
  __syncthreads();

  // Generate partial histogram
  for(int pos = globalTid; pos < len; pos += numThreads) {
    atomicAdd(&partialHistogram[input[pos]], 1);
  }
  __syncthreads();

  // Merge partial histograms
  for(int pos = threadIdx.x; pos < HISTOGRAM_LENGTH; pos += blockDim.x) {
    atomicAdd(output + pos, partialHistogram[pos]);
  }
}

// Sequential histogram
// __global__ void histogram(unsigned char *input, uint *output, int len) {
//   for(int i = 0; i < HISTOGRAM_LENGTH; ++i) {
//     output[i] = 0;
//   }
//   for(int i = 0; i < len; ++i) {
//     output[input[i]]++;
//   }
// }

// Compute CDF of histogram
// __global__ void cdf(uint *histogram, float *output, int width, int height) {
//   output[0] = (float)histogram[0] / (float)(width * height);
//   for(int i = 1; i < HISTOGRAM_LENGTH; ++i) {
//     output[i] = output[i-1] + (float)histogram[i] / (float)(width * height);
//   }
// }

// Sequential CDF
__global__ void cdf(uint *histogram, float *output, int width, int height) {
  output[0] = (float)histogram[0] / (float)(width * height);
  for(int i = 1; i < 256; ++i) {
    output[i] = output[i-1] + (float)histogram[i] / (float)(width * height);
  }
}

__device__ float clamp(float x, float start, float end) {
  return min(max(x, start), end);
}

__device__ unsigned char correct_color(float *cdf, unsigned char val) {
  float temp = 255.0f * (cdf[val] - cdf[0]) / (1.0 - cdf[0]);
  if(temp < 0) {
    return 0;
  }
  if(temp > 255) {
    return 255;
  }
  return (unsigned char)temp;
}


// __global__ void equalize(unsigned char *ucharImage, float *cdf, unsigned char *output, int width, int height, int channels) {
//   int tx = (blockIdx.x * blockDim.x + threadIdx.x)*3;
//   if(tx < width * height * channels) {
//     // output[tx] = correct_color(cdf, ucharImage[tx]);
//   }
// }

// Sequential equalizer
__global__ void equalize(unsigned char *ucharImage, float *cdf, unsigned char *output, int width, int height, int channels) {
  for(int i = 0; i < width * height * channels; ++i) {
    output[i] = correct_color(cdf, ucharImage[i]);
  }
}

// Cast the image from float to unsigned char
__global__ void castf(unsigned char *input, float *output, int width, int height, int channels) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  if(tx < width * height * channels) {
    output[tx] = (float)(input[tx] / 255.0f);
  }
}

// Sequential castf
// __global__ void castf(unsigned char *input, float *output, int width, int height, int channels) {
//   for(int i = 0; i < width * height * channels; ++i) {
//     output[i] = (float)(input[i] / 255.0f);
//   }
// }

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  int imageElts;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;
  
  //@@ Insert more code here
  
  // Cast unsigned char kernel
  // float *castucHostIn;
  float *castucDeviceIn;
  unsigned char *castucDeviceOut;

  // Convert kernel
  unsigned char *convertHostIn;
  unsigned char *convertDeviceIn;
  unsigned char *convertDeviceOut;

  // Histogram kernel
  unsigned char *histHostIn;
  unsigned char *histDeviceIn;
  uint *histDeviceOut;

  // CDF kernel
  uint *cdfHostIn;
  uint *cdfDeviceIn;
  float *cdfDeviceOut;

  // Equalize kernel
  float *eqHostIn;
  float *eqDeviceIn;
  unsigned char *eqDeviceOut;

  // Cast float kernel
  unsigned char *castfHostIn;
  unsigned char *castfDeviceIn;
  float *castfDeviceOut;

  float *castfHostOut;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  imageElts = imageWidth * imageHeight * imageChannels;
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  hostInputImageData = wbImage_getData(inputImage);
  // hostOutputImageData = wbImage_getData(outputImage);

  // wbLog(TRACE, "Width: ", imageWidth);
  // wbLog(TRACE, "Height: ", imageHeight);
  // wbLog(TRACE, "Channels: ", imageChannels);

  // wbLog(TRACE, "hostInputImageData: ", hostInputImageData[0]);
  // for(int i = 0; i < 10; ++i) {
  //   wbLog(TRACE, "hostInputImageData ", (float)hostInputImageData[i]);
  // }

  //-------------------------------------- Cast uc kernel -------------------------
  // Allocate CPU and GPU memory
  // castucHostIn = (float *)malloc(imageElts * sizeof(float));
  cudaMalloc(&castucDeviceIn, imageElts * sizeof(float));
  cudaMalloc(&castucDeviceOut, imageElts * sizeof(unsigned char));
  convertHostIn = (unsigned char *)malloc(imageElts * sizeof(unsigned char));
  // Copy memory to GPU
  cudaMemcpy(castucDeviceIn, hostInputImageData, imageElts * sizeof(float), cudaMemcpyHostToDevice);
  // Initialize block and grid dimensions
  dim3 blocksPerGrid1(ceil((float)imageChannels * (float)imageHeight * (float)imageWidth / (float)BLOCK_SIZE), 1, 1);
  dim3 threadsPerBlock1(BLOCK_SIZE, 1, 1);
  // Launch kernel
  castuc<<<blocksPerGrid1, threadsPerBlock1>>>(castucDeviceIn, castucDeviceOut, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();
  // Copy result from kernel back to host
  cudaMemcpy(convertHostIn, castucDeviceOut, imageElts * sizeof(unsigned char), cudaMemcpyDeviceToHost);
  // Free everything except result in host
  // free(castucHostIn);
  cudaFree(castucDeviceIn);
  cudaFree(castucDeviceOut);
  //-------------------------------------- Cast uc kernel -------------------------
  // for(int i = 0; i < 10; ++i) {
  //   wbLog(TRACE, "ucharImage: ", (float)convertHostIn[i]);
  // }

  //-------------------------------------- Convert kernel -------------------------
  // Allocate CPU and GPU memory
  // convertHostIn = (unsigned char *)malloc(imageElts * sizeof(unsigned char));
  cudaMalloc(&convertDeviceIn, imageElts * sizeof(unsigned char));
  cudaMalloc(&convertDeviceOut, imageWidth * imageHeight * sizeof(unsigned char));
  histHostIn = (unsigned char *)malloc(imageWidth * imageHeight * sizeof(unsigned char));
  // Copy memory to GPU
  cudaMemcpy(convertDeviceIn, convertHostIn, imageElts * sizeof(unsigned char), cudaMemcpyHostToDevice);
  // Initialize block and grid dimensions
  dim3 blocksPerGrid2(1, 1, 1);
  dim3 threadsPerBlock2(1, 1, 1);
  // Launch kernel
  convert<<<blocksPerGrid2, threadsPerBlock2>>>(convertDeviceIn, convertDeviceOut, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();
  // Copy result from kernel back to host
  cudaMemcpy(histHostIn, convertDeviceOut, imageWidth * imageHeight * sizeof(unsigned char), cudaMemcpyDeviceToHost);
  // Free everything except result in host
  // free(convertHostIn);
  cudaFree(convertDeviceIn);
  cudaFree(convertDeviceOut);
  //-------------------------------------- Convert kernel -------------------------
  // for(int i = 0; i < 10; ++i) {
  //   wbLog(TRACE, "grayImage: ", (float)histHostIn[i]);
  // }

  //-------------------------------------- Histogram kernel -------------------------
  // Allocate CPU and GPU memory
  // histHostIn = (uint *)malloc(imageElts * sizeof(unsigned char));
  cudaMalloc(&histDeviceIn, imageWidth * imageHeight * sizeof(unsigned char));
  cudaMalloc(&histDeviceOut, HISTOGRAM_LENGTH * sizeof(uint));
  cdfHostIn = (uint *)malloc(HISTOGRAM_LENGTH * sizeof(uint));
  // Copy memory to GPU
  cudaMemcpy(histDeviceIn, histHostIn, imageWidth * imageHeight * sizeof(unsigned char), cudaMemcpyHostToDevice);
  // Initialize block and grid dimensions
  dim3 blocksPerGrid3(ceil((float)imageWidth * (float)imageHeight / (float)BLOCK_SIZE), 1, 1);
  dim3 threadsPerBlock3(BLOCK_SIZE, 1, 1);
  // Launch kernel
  histogram<<<blocksPerGrid3, threadsPerBlock3>>>(histDeviceIn, histDeviceOut, imageWidth * imageHeight);
  // histogram<<<1, 256>>>(histDeviceIn, histDeviceOut, imageWidth * imageHeight);

  cudaDeviceSynchronize();
  // Copy result from kernel back to host
  cudaMemcpy(cdfHostIn, histDeviceOut, HISTOGRAM_LENGTH * sizeof(uint), cudaMemcpyDeviceToHost);
  // Free everything except result in host
  free(histHostIn);
  cudaFree(histDeviceIn);
  cudaFree(histDeviceOut);
  //-------------------------------------- Histogram kernel -------------------------
  // int sum = 0;
  // for(int i = 0; i < 256; ++i) {
  //   wbLog(TRACE, "Histogram: ", cdfHostIn[i]);
  //   sum += cdfHostIn[i];
  // }
  // wbLog(TRACE, "Histogram total: ", sum);

  //-------------------------------------- CDF kernel -------------------------
  // Allocate CPU and GPU memory
  // histHostIn = (uint *)malloc(imageElts * sizeof(unsigned char));
  cudaMalloc(&cdfDeviceIn, HISTOGRAM_LENGTH * sizeof(uint));
  cudaMalloc(&cdfDeviceOut, HISTOGRAM_LENGTH * sizeof(float));
  eqHostIn = (float *)malloc(HISTOGRAM_LENGTH * sizeof(float));
  // Copy memory to GPU
  cudaMemcpy(cdfDeviceIn, cdfHostIn, HISTOGRAM_LENGTH * sizeof(int), cudaMemcpyHostToDevice);
  // Initialize block and grid dimensions
  // dim3 blocksPerGrid4(ceil((float)imageChannels * (float)imageHeight * (float)imageWidth / (float)BLOCK_SIZE), 1, 1);
  // dim3 threadsPerBlock4(BLOCK_SIZE, 1, 1);
  // Launch kernel
  cdf<<<1, 1>>>(cdfDeviceIn, cdfDeviceOut, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  // Copy result from kernel back to host
  cudaMemcpy(eqHostIn, cdfDeviceOut, HISTOGRAM_LENGTH * sizeof(float), cudaMemcpyDeviceToHost);
  // Free everything except result in host
  free(cdfHostIn);
  cudaFree(cdfDeviceIn);
  cudaFree(cdfDeviceOut);
  //-------------------------------------- CDF kernel -------------------------
  
  for(int i = 0; i < 10; ++i) {
    wbLog(TRACE, "ucharImage: ", (float)convertHostIn[i]);
  }
  // for(int i = 0; i < 256; ++i) {
  //   wbLog(TRACE, "CDF ", i, ": ", eqHostIn[i]);
  // }

  //-------------------------------------- Equalizer kernel -------------------------
  // Allocate CPU and GPU memory
  // histHostIn = (uint *)malloc(imageElts * sizeof(unsigned char));
  cudaMalloc(&eqDeviceIn, HISTOGRAM_LENGTH * sizeof(float));
  cudaMalloc(&eqDeviceOut, imageElts * sizeof(unsigned char));
  castfHostIn = (unsigned char *)malloc(imageElts * sizeof(unsigned char));
  // Copy memory to GPU
  cudaMemcpy(eqDeviceIn, eqHostIn, HISTOGRAM_LENGTH * sizeof(float), cudaMemcpyHostToDevice);
  // Initialize block and grid dimensions
  // dim3 blocksPerGrid5(ceil((float)imageChannels * (float)imageHeight * (float)imageWidth / (float)BLOCK_SIZE), 1, 1);
  // dim3 threadsPerBlock5(BLOCK_SIZE, 1, 1);
  // Launch kernel
  equalize<<<1, 1>>>(convertHostIn, eqDeviceIn, eqDeviceOut, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();
  // Copy result from kernel back to host
  cudaMemcpy(castfHostIn, eqDeviceOut,  imageElts * sizeof(unsigned char), cudaMemcpyDeviceToHost);
  // Free everything except result in host
  free(eqHostIn);
  cudaFree(eqDeviceIn);
  cudaFree(eqDeviceOut);
  //-------------------------------------- Equalizer kernel -------------------------
  for(int i = 0; i < imageElts; ++i) {
    float temp = 255.0f * (eqHostIn[convertHostIn[i]] - eqHostIn[0]) / (1.0 - eqHostIn[0]);
    if(temp < 0) {
      return 0;
    }
    if(temp > 255) {
      return 255;
    }
    castfHostIn[i] = temp;
  }
  
  // for(int i = 0; i < 10; ++i) {
  //   wbLog(TRACE, "castfHostIn ", (float)castfHostIn[i]);
  // }

  // ==========================================================================================================
/*
  //-------------------------------------- Equalized Convert kernel -------------------------
  // Allocate CPU and GPU memory
  // convertHostIn = (unsigned char *)malloc(imageElts * sizeof(unsigned char));
  cudaMalloc(&convertDeviceIn, imageElts * sizeof(unsigned char));
  cudaMalloc(&convertDeviceOut, imageWidth * imageHeight * sizeof(unsigned char));
  histHostIn = (unsigned char *)malloc(imageWidth * imageHeight * sizeof(unsigned char));
  // Copy memory to GPU
  cudaMemcpy(convertDeviceIn, castfHostIn, imageElts * sizeof(unsigned char), cudaMemcpyHostToDevice);
  // Initialize block and grid dimensions
  // dim3 blocksPerGrid(ceil((float)imageChannels * (float)imageHeight * (float)imageWidth / (float)BLOCK_SIZE), 1, 1);
  // dim3 threadsPerBlock(BLOCK_SIZE, 1, 1);
  // Launch kernel
  convert<<<1, 1>>>(convertDeviceIn, convertDeviceOut, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();
  // Copy result from kernel back to host
  cudaMemcpy(histHostIn, convertDeviceOut, imageWidth * imageHeight * sizeof(unsigned char), cudaMemcpyDeviceToHost);
  // Free everything except result in host
  // free(convertHostIn);
  cudaFree(convertDeviceIn);
  cudaFree(convertDeviceOut);
  //-------------------------------------- Equalized Convert kernel -------------------------

  //-------------------------------------- Equalized Histogram kernel -------------------------
  // Allocate CPU and GPU memory
  // histHostIn = (uint *)malloc(imageElts * sizeof(unsigned char));
  cudaMalloc(&histDeviceIn, imageWidth * imageHeight * sizeof(unsigned char));
  cudaMalloc(&histDeviceOut, HISTOGRAM_LENGTH * sizeof(uint));
  cdfHostIn = (uint *)malloc(HISTOGRAM_LENGTH * sizeof(uint));
  // Copy memory to GPU
  cudaMemcpy(histDeviceIn, histHostIn, imageWidth * imageHeight * sizeof(unsigned char), cudaMemcpyHostToDevice);
  // Initialize block and grid dimensions
  // dim3 blocksPerGrid(ceil((float)imageChannels * (float)imageHeight * (float)imageWidth / (float)BLOCK_SIZE), 1, 1);
  // dim3 threadsPerBlock(BLOCK_SIZE, 1, 1);
  // Launch kernel
  // histogram<<<blocksPerGrid, threadsPerBlock>>>(histDeviceIn, histDeviceOut, imageElts);
  histogram<<<1, 1>>>(histDeviceIn, histDeviceOut, imageWidth * imageHeight);

  cudaDeviceSynchronize();
  // Copy result from kernel back to host
  cudaMemcpy(cdfHostIn, histDeviceOut, HISTOGRAM_LENGTH * sizeof(uint), cudaMemcpyDeviceToHost);
  // Free everything except result in host
  free(histHostIn);
  cudaFree(histDeviceIn);
  cudaFree(histDeviceOut);
  //-------------------------------------- Equalized Histogram kernel -------------------------

  //-------------------------------------- Equalized CDF kernel -------------------------
  // Allocate CPU and GPU memory
  // histHostIn = (uint *)malloc(imageElts * sizeof(unsigned char));
  cudaMalloc(&cdfDeviceIn, HISTOGRAM_LENGTH * sizeof(uint));
  cudaMalloc(&cdfDeviceOut, HISTOGRAM_LENGTH * sizeof(float));
  eqHostIn = (float *)malloc(HISTOGRAM_LENGTH * sizeof(float));
  // Copy memory to GPU
  cudaMemcpy(cdfDeviceIn, cdfHostIn, HISTOGRAM_LENGTH * sizeof(int), cudaMemcpyHostToDevice);
  // Initialize block and grid dimensions
  // dim3 blocksPerGrid(ceil((float)imageChannels * (float)imageHeight * (float)imageWidth / (float)BLOCK_SIZE), 1, 1);
  // dim3 threadsPerBlock(BLOCK_SIZE, 1, 1);
  // Launch kernel
  cdf<<<1, 1>>>(cdfDeviceIn, cdfDeviceOut, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  // Copy result from kernel back to host
  cudaMemcpy(eqHostIn, cdfDeviceOut, HISTOGRAM_LENGTH * sizeof(float), cudaMemcpyDeviceToHost);
  // Free everything except result in host
  free(cdfHostIn);
  cudaFree(cdfDeviceIn);
  cudaFree(cdfDeviceOut);
  //-------------------------------------- Equalized CDF kernel -------------------------
  for(int i = 0; i < 256; ++i) {
    wbLog(TRACE, "CDF ", i, ": ", eqHostIn[i]);
  }
*/

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  castfHostOut = (float *)malloc(imageElts * sizeof(float));
  hostOutputImageData = (float *)malloc(imageElts * sizeof(float));
  for(int i = 0; i < imageElts; ++i) {
    hostOutputImageData[i] = (float)castfHostIn[i] / 255.0f;
  }

  // for(int i = 0; i < 10; ++i) {
  //   wbLog(TRACE, "castfHostOut ", castfHostOut[i]);
  // }
  //-------------------------------------- Cast f kernel -------------------------
  // Allocate CPU and GPU memory
  // histHostIn = (uint *)malloc(imageElts * sizeof(unsigned char));
  cudaMalloc(&castfDeviceIn, imageElts * sizeof(unsigned char));
  cudaMalloc(&castfDeviceOut, imageElts * sizeof(float));
  // Copy memory to GPU
  cudaMemcpy(castfDeviceIn, castfHostIn, imageElts * sizeof(unsigned char), cudaMemcpyHostToDevice);
  // Initialize block and grid dimensions
  dim3 blocksPerGrid6(ceil((float)imageElts / (float)BLOCK_SIZE), 1, 1);
  dim3 threadsPerBlock6(BLOCK_SIZE, 1, 1);
  // Launch kernel
  // castf<<<blocksPerGrid6, threadsPerBlock6>>>(castfDeviceIn, castfDeviceOut, imageWidth, imageHeight, imageChannels);
  // castf<<<1, 1>>>(castfDeviceIn, castfDeviceOut, imageWidth, imageHeight, imageChannels);

  cudaDeviceSynchronize();
  // Copy result from kernel back to host
  cudaMemcpy(hostOutputImageData, castfDeviceOut, imageElts * sizeof(float), cudaMemcpyDeviceToHost);
  // Free everything except result in host
  free(castfHostIn);
  cudaFree(castfDeviceIn);
  cudaFree(castfDeviceOut);
  //-------------------------------------- Cast f kernel -------------------------

  // for(int i = 0; i < 10; ++i) {
  //   wbLog(TRACE, "hostOutputImageData ", hostOutputImageData[i]);
  // }

  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels, hostOutputImageData);
  wbSolution(args, outputImage);

  //@@ insert code here
  free(inputImage);
  free(outputImage);
  free(hostInputImageData);
  free(hostOutputImageData);
  // free(inputImageFile);
  return 0;
}
