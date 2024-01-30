# WinoFPGA
本项目基于[CFU-Playground](https://github.com/google/CFU-Playground)进行开发，设计了一个基于RISC-V的轻量化深度可分离卷积神经网络加速器，旨在弥补RISC-V处理器的卷积计算能力的不足.该加速器支持深度可分离卷积中的两个关键算子，即深度卷积和点卷积，并能够通过共享硬件结构提高资源利用效率.深度卷积计算流水线采用了高效的Winograd卷积算法，并使用2×2数据块组合拼接成4×4数据片的方式来减少传输数据冗余.同时，通过拓展RISC-V处理器端指令，使得加速器能够实现更灵活的配置和调用.运行结果表明，相较于基础的RISC-V处理器，调用加速器后的点卷积和深度卷积计算取得了显著的加速效果，其中点卷积加速了104.40倍，深度卷积加速了123.63倍.与此同时，加速器的性能功耗比达到了8.7GOPS/W.
## 硬件平台
开发设备：联想ThinkBook14（AMD7840H，内存32G，硬盘1T）  
加速器实现设备：Arty-A7（搭载Xilinx公司的XC7A100T芯片）
## 软件情况
开发系统：Ubuntu 18.04（运行于VMware Workstation 16 Player）  
开发软件：Visual Studio Code 1.74.3、Vivado 2022
## 使用方法
1. 环境配置
参考[CFU-Playground](https://github.com/google/CFU-Playground)相关说明
2. 
