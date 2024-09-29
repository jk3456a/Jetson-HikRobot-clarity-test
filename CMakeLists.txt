cmake_minimum_required(VERSION 3.10)
project(my_test LANGUAGES CXX CUDA)  # 添加CUDA作为项目语言

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Release) 

# 检测平台
if (WIN32)
    set(MVS_SDK_PATH "C:/Program Files (x86)/MVS/Development")
    include_directories(${MVS_SDK_PATH}/Includes)
    link_directories(${MVS_SDK_PATH}/Libraries/win64)
else()
    set(MVS_SDK_PATH "/opt/MVS")
    include_directories(${MVS_SDK_PATH}/include)
    link_directories(${MVS_SDK_PATH}/lib/aarch64)
endif()

# setup CUDA
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)

    # to tell your code it was found via #ifdef USE_CUDA:
    add_definitions(-DUSE_CUDA)    

    include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
endif()

# 查找源码文件
file(GLOB SOURCES "src/*.cpp")
file(GLOB CUDA_SOURCES "src/*.cu")  # 查找CUDA源文件
file(GLOB_RECURSE HEADERS "include/*.h")
include_directories(include)
# 指定生成可执行文件
add_executable(${PROJECT_NAME} ${SOURCES} ${CUDA_SOURCES} ${HEADERS})

# 链接依赖库
if(WIN32)
    target_link_libraries(${PROJECT_NAME} MvCameraControl.lib)  # Windows上的库文件
else()
    target_link_libraries(${PROJECT_NAME} MvCameraControl)  # Linux上的库文件
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "87")
