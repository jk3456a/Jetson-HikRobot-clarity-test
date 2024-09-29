#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define checkCudaErrors(call)                                 \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)
  
// #define CHECK(call) \
// do                  \
// {\
//     const cudaError_t error_code = call; \
//     if(error_code != cudaSuccess) \   
//     {\
//         printf("cuda error"); \
//         printf("  File:   %s\n",__FILE__); \
//         printf("  Line:   %d\n",__LINE__); \
//         printf("  Error code: %d\n",error_code); \
//         printf("  Error text: %s\n",cudaGetErrorString(error_code)); \
//         exit(1); \
//     }\
// } while(0)

#define BLOCKDIM_X 16
#define BLOCKDIM_Y 16
#define BLOCKDIM_X_SM20 32
#define BLOCKDIM_Y_SM20 32
#define BLOCK_SIZE 1024
inline int iDivUp(int a, int b);
__global__ void cuBrenner(ulong2 *inData, ulong2 *outData, unsigned int n);
__global__ void cuBrennerInit(unsigned char* pGray, ulong2* pTmp, const int width, const int height, const int gridWidth, const int numBlocks);
__global__ void cuBrennerInitWH(unsigned char* pGray, ulong2* pTmp, const int width, const int height, const int gridWidth, const int numBlocks);
bool RunCuBrenner(unsigned char* pGray, ulong2* pTmp, const int width, const int height, const int numSMs, int version, int brennerType);