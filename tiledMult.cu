
#include <wb.h>
#include "support.h"

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
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
  Timer timer;
  cudaError_t cuda_ret;

  args = wbArg_read(argc, argv);

  // Initialize host variables ----------------------------------------------

  printf("\nImporting data and creating memory on host..."); fflush(stdout);
  startTime(&timer);
  //wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
                            
  //@@ Set numCRows and numCColumns (to something other than 0, obviously)
  numCRows = 0;
  numCColumns = 0;
  //@@ Allocate the hostC matrix
  
  stopTime(&timer); printf("%f s\n", elapsedTime(timer));
  //wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  printf("Allocating GPU memory..."); fflush(stdout);
  startTime(&timer);
  //wbTime_start(GPU, "Allocating GPU memory.");
  
  //@@ Allocate GPU memory here

  stopTime(&timer); printf("%f s\n", elapsedTime(timer));
  //wbTime_stop(GPU, "Allocating GPU memory.");

  printf("Copying input memory to the GPU..."); fflush(stdout);
  startTime(&timer);
  //wbTime_start(GPU, "Copying input memory to the GPU.");
  
  //@@ Copy memory to the GPU here

  stopTime(&timer); printf("%f s\n", elapsedTime(timer));
  //wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here

  printf("Performing CUDA computation..."); fflush(stdout);
  startTime(&timer);
  //wbTime_start(Compute, "Performing CUDA computation");
  
  //@@ Launch the GPU Kernel here

  cudaDeviceSynchronize();
  stopTime(&timer); printf("%f s\n", elapsedTime(timer));
  //wbTime_stop(Compute, "Performing CUDA computation");

  printf("Copying output memory to the CPU..."); fflush(stdout);
  startTime(&timer);
  //wbTime_start(Copy, "Copying output memory to the CPU");
  
  //@@ Copy the GPU memory back to the CPU here

  stopTime(&timer); printf("%f s\n", elapsedTime(timer));
  //wbTime_stop(Copy, "Copying output memory to the CPU");

  printf("Freeing GPU Memory.."); fflush(stdout);
  startTime(&timer);
  //wbTime_start(GPU, "Freeing GPU Memory");
  
  //@@ Free the GPU memory here

  stopTime(&timer); printf("%f s\n", elapsedTime(timer));
  //wbTime_stop(GPU, "Freeing GPU Memory");

  //Determine if output is correct and print result.
  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
