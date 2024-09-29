# jetson-HikRobot-clarity-test

## 0 运行环境

开发板：jetson orin nx 8GB
开发环境：Jetpack 6.0 + Jetson Linux 36.3 
cuda版本：12.2
开发板算力架构: 87

## 1 简述

jetson与海康工业相机连接并取图到显存中计算的简单的测试demo，本次测试主要关注于一下方面：
1. 一致性内存与分别存储的性能差异 通过修改#define USE_UNIFIED_MEMORY 的值来确定是否使用一致性内存，我们的使用场景下，使用一致性内存有30%的时间上的提升，同时gpu占用率下降50%
2. gpu不同频率下的性能差异（`sudo jetson_clocks`可以直接以最大频率运行，正常的修改gpu频率的逻辑与修改cpu的一致）

代码功能：从MVSDK中获得图片到显存中计算清晰度，再从cpu中读取并计算清晰度

## 2 运行方式

```bash
cd build
cmake ..
make
```
一般情况下可以正常运行
