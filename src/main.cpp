#include <iostream>
#include <fstream>
#include <cstring>
#include <chrono>
#include "MvCameraControl.h"
#include "Clarity.h"
#define USE_UNIFIED_MEMORY 1

void PrintDeviceInfo(MV_CC_DEVICE_INFO* pstMVDevInfo)
{
    if (NULL == pstMVDevInfo)
    {
        printf("The Pointer of pstMVDevInfo is NULL!\n");
        return;
    }
    if (pstMVDevInfo->nTLayerType == MV_GIGE_DEVICE)
    {
        int nIp1 = ((pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24);
        int nIp2 = ((pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16);
        int nIp3 = ((pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8);
        int nIp4 = (pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff);

        printf("Device Model Name: %s\n", pstMVDevInfo->SpecialInfo.stGigEInfo.chModelName);
        printf("CurrentIp: %d.%d.%d.%d\n", nIp1, nIp2, nIp3, nIp4);
        printf("UserDefinedName: %s\n\n", pstMVDevInfo->SpecialInfo.stGigEInfo.chUserDefinedName);
    }
    else if (pstMVDevInfo->nTLayerType == MV_USB_DEVICE)
    {
        printf("Device Model Name: %s\n", pstMVDevInfo->SpecialInfo.stUsb3VInfo.chModelName);
        printf("UserDefinedName: %s\n\n", pstMVDevInfo->SpecialInfo.stUsb3VInfo.chUserDefinedName);
    }
}

int main()
{
#ifdef LOG_TYPE
    std::string logPath = "..\\data\\log.txt";
    std::ofstream logFile(logPath);

    //logfile set
    if (!logFile) {
        std::cerr << "Error: Could not open log file." << std::endl;
        return 1;
    }
#endif
    // struct __device_builtin__ __align__(2*sizeof(long int)) long2
    // {
    //     long int x, y;
    // };

    // struct __device_builtin__ __align__(2*sizeof(unsigned long int)) ulong2
    // {
    //     unsigned long int x, y;
    // };
    //cuda init
    int cuda_device_count;
    cudaGetDeviceCount(&cuda_device_count); // 获取可用设备数量
    cudaDeviceProp deviceProp;
    for (int i = 0; i < cuda_device_count; ++i) {
        cudaGetDeviceProperties(&deviceProp, i); // 获取设备属性
        // 这里可以使用 deviceProp 进行其他操作
        std::cout << "使用GPU：" << i << std::endl;
		std::cout << "设备全局内存总量：" << deviceProp.totalGlobalMem << std::endl;
		std::cout << "SM的数量:" << deviceProp.multiProcessorCount << std::endl;
		std::cout << "每个线程块的共享内存大小：" << deviceProp.sharedMemPerBlock / 1024<< "KB" << std::endl;
		std::cout << "每个线程块的最大线程数：" << deviceProp.maxThreadsPerBlock << std::endl;
		std::cout << "一个线程块中可用的寄存器数：" << deviceProp.regsPerBlock << std::endl;
		std::cout << "每个EM的最大线程数：" << deviceProp.maxThreadsPerMultiProcessor << std::endl;
		std::cout << "每个EM的最大线束数：" << deviceProp.maxThreadsPerMultiProcessor / 32<< std::endl;
		std::cout << "设备上多处理器的数量：" << deviceProp.multiProcessorCount << std::endl;

    }
    int gpuNums = deviceProp.multiProcessorCount;
    int gpuVersion = deviceProp.major * 10 + deviceProp.minor;
    //camera set
    int nRet = MV_OK;

    void* handle = nullptr;
    MV_CC_DEVICE_INFO_LIST stDeviceList;
    memset(&stDeviceList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));

    // 枚举设备
    // enum device
    nRet = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &stDeviceList);
    if (MV_OK != nRet)
    {
        printf("MV_CC_EnumDevices fail! nRet [%x]\n", nRet);
        return -1;
    }

    if (stDeviceList.nDeviceNum > 0)
    {
        for (int i = 0; i < stDeviceList.nDeviceNum; i++)
        {
            printf("[device %d]:\n", i);
            MV_CC_DEVICE_INFO* pDeviceInfo = stDeviceList.pDeviceInfo[i];
            if (NULL == pDeviceInfo)
            {
                break;
            } 
            PrintDeviceInfo(pDeviceInfo);            
        }  
    } 
    else
    {
        printf("Find No Devices!\n");
        return -1;
    }

    unsigned int nIndex = 0;

    // 选择设备并创建句柄
    // select device and create handle
    nRet = MV_CC_CreateHandle(&handle, stDeviceList.pDeviceInfo[nIndex]);
    if (MV_OK != nRet)
    {
        printf("MV_CC_CreateHandle fail! nRet [%x]\n", nRet);
        return -1;
    }

    // 打开设备
    // open device
    nRet = MV_CC_OpenDevice(handle);
    if (MV_OK != nRet)
    {
        printf("MV_CC_OpenDevice fail! nRet [%x]\n", nRet);
        return -1;
    }

    // 设置图像格式 保留bayar todo
    MV_CC_SetPixelFormat(handle, PixelType_Gvsp_BayerGR8);
    // 关闭自动曝光
    MV_CC_SetEnumValue(handle, "ExposureMode", MV_EXPOSURE_MODE_TIMED);
    MV_CC_SetEnumValue(handle, "ExposureTimeMode", 1);
    MV_CC_SetEnumValue(handle, "ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF);
    MV_CC_SetFloatValue(handle, "ExposureTime", 10.0f);
    // 设置触发模式为off
    // set trigger mode as off
    nRet = MV_CC_SetEnumValue(handle, "TriggerMode", 0);
    // 设置窗口大小
    MV_CC_SetIntValueEx(handle, "Width", 1440);
    MV_CC_SetIntValueEx(handle, "Height", 80);
    MV_CC_SetFloatValue(handle, "AcquisitionFrameRate", 2500.0f);
    // 开始取流
    // start grab image
    nRet = MV_CC_StartGrabbing(handle);
    if (MV_OK != nRet)
    {
        printf("MV_CC_StartGrabbing fail! nRet [%x]\n", nRet);
        return -1;
    }

    // 分配存图内存和显存
    MVCC_INTVALUE stParam;
    memset(&stParam, 0, sizeof(MVCC_INTVALUE));
    nRet = MV_CC_GetIntValue(handle, "PayloadSize", &stParam);
    if (MV_OK != nRet)
    {
        printf("Get PayloadSize fail! nRet [0x%x]\n", nRet);
        return nRet;
    }
    unsigned int nDataSize = stParam.nCurValue;
    unsigned char* pData;
#if USE_UNIFIED_MEMORY    
    cudaError_t err = cudaMallocManaged(&pData, sizeof(unsigned char) * nDataSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        MV_CC_StopGrabbing(handle);
        MV_CC_DestroyHandle(handle);
        return -1;
    }
#else
    pData = (unsigned char*)malloc(sizeof(unsigned char) * nDataSize);
    if (pData == nullptr)
    {
        std::cerr << "Allocate memory fail!" << std::endl;
        MV_CC_StopGrabbing(handle);
        MV_CC_DestroyHandle(handle);
        return -1;
    }
    unsigned char* d_pData;
    if (cudaMalloc((void**)&d_pData, nDataSize) != cudaSuccess) {
        std::cerr << "Error: cudaMalloc failed!" << std::endl;
        return -1;  
    }
#endif


    ulong2 * clarity_sum = NULL; // 清晰度合计中间变量
#if USE_UNIFIED_MEMORY
    if (cudaMallocManaged(&clarity_sum, nDataSize * sizeof(ulong2)) != cudaSuccess) {
        std::cerr << "Error: cudaMallocManaged failed!" << std::endl;
        return -1;  
    }
#else
    clarity_sum = (ulong2*)malloc(nDataSize * sizeof(ulong2));
    if (clarity_sum == nullptr)
    {
        std::cerr << "Allocate memory fail!" << std::endl;
        return -1;
    }
    ulong2 * d_clarity_sum;
    if (cudaMalloc((void**)&d_clarity_sum, nDataSize * sizeof(ulong2)) != cudaSuccess) {
        std::cerr << "Error: cudaMalloc failed!" << std::endl;
        return -1;  
    }
#endif

    MV_FRAME_OUT_INFO_EX stImageInfo = {0};
    int frameCount = 0;
    while (true)
    {   
        nRet = MV_CC_GetOneFrameTimeout(handle, pData, nDataSize , &stImageInfo, 500);
        if (MV_OK != nRet)
        {
             std::cerr << "No frame data received. nRet [" << std::hex << nRet << "]" << std::endl;
        }
        bool isMono;
        switch (stImageInfo.enPixelType)
        //switch (stImageInfo.stFrameInfo.enPixelType)
        {
        case PixelType_Gvsp_Mono8:
        case PixelType_Gvsp_Mono10:
        case PixelType_Gvsp_Mono10_Packed:
        case PixelType_Gvsp_Mono12:
        case PixelType_Gvsp_Mono12_Packed:
            isMono=true;
            break;
        default:
            isMono=false;
            break;
        }

        if(isMono){
            auto start_time = std::chrono::high_resolution_clock::now();
            auto start_time_micro = std::chrono::time_point_cast<std::chrono::microseconds>(start_time);
#if USE_UNIFIED_MEMORY
            RunCuBrenner(pData, clarity_sum, stImageInfo.nWidth, stImageInfo.nHeight, gpuNums, gpuVersion, 3);
#else
            cudaMemcpy(d_pData, pData, nDataSize, cudaMemcpyHostToDevice);
            RunCuBrenner(d_pData, d_clarity_sum, stImageInfo.nWidth, stImageInfo.nHeight, gpuNums, gpuVersion, 3);
            checkCudaErrors(cudaDeviceSynchronize());
            cudaMemcpy(clarity_sum, d_clarity_sum, nDataSize * sizeof(ulong2), cudaMemcpyDeviceToHost);
#endif
            float clarity = (clarity_sum->x==0)?.0f: (100.0f * clarity_sum->y / clarity_sum->x);
            auto end_time = std::chrono::high_resolution_clock::now();
            auto end_time_micro = std::chrono::time_point_cast<std::chrono::microseconds>(end_time);
            auto duration = end_time_micro.time_since_epoch().count() - start_time_micro.time_since_epoch().count();
            std::cout<< "frame :"<<frameCount<<" clarity: "<<clarity<<"calculate time:"<<duration<<std::endl;
            frameCount++;  
        }
        
    }
        
    nRet = MV_CC_StopGrabbing(handle);
    if (MV_OK != nRet)
    {
        printf("MV_CC_StopGrabbing fail! nRet [%x]\n", nRet);
    }

    // 关闭设备
    // close device
    nRet = MV_CC_CloseDevice(handle);
    if (MV_OK != nRet)
    {
        printf("MV_CC_CloseDevice fail! nRet [%x]\n", nRet);
    }

    // 销毁句柄
    // destroy handle
    nRet = MV_CC_DestroyHandle(handle);
    if (MV_OK != nRet)
    {
        printf("MV_CC_DestroyHandle fail! nRet [%x]\n", nRet);
    }

#ifdef USE_UNIFIED_MEMORY
    cudaFree(pData);
    cudaFree(clarity_sum);
#else   
    free(pData);   
    free(clarity_sum);
    cudaFree(d_pData);
    cudaFree(d_clarity_sum);
#endif
    return 0;
}
