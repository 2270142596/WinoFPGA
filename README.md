# WinoFPGA
本项目基于[CFU-Playground](https://github.com/google/CFU-Playground)进行开发，设计了一个基于RISC-V的轻量化深度可分离卷积神经网络加速器，旨在弥补RISC-V处理器的卷积计算能力的不足.该加速器支持深度可分离卷积中的两个关键算子，即深度卷积和点卷积，并能够通过共享硬件结构提高资源利用效率.深度卷积计算流水线采用了高效的Winograd卷积算法，并使用2×2数据块组合拼接成4×4数据片的方式来减少传输数据冗余.同时，通过拓展RISC-V处理器端指令，使得加速器能够实现更灵活的配置和调用.运行结果表明，相较于基础的RISC-V处理器，调用加速器后的点卷积和深度卷积计算取得了显著的加速效果，其中点卷积加速了104.40倍，深度卷积加速了123.63倍.与此同时，加速器的性能功耗比达到了8.7GOPS/W.系统架构如下：
![image](https://github.com/2270142596/WinoFPGA/blob/master/picture/arch.png)  
## 硬件平台
开发设备：联想ThinkBook14（AMD7840H，内存32G，硬盘1T）  
加速器实现设备：Arty-A7（搭载Xilinx公司的XC7A100T芯片）
## 软件情况
开发系统：Ubuntu 18.04（运行于VMware Workstation 16 Player）  
开发软件：Visual Studio Code 1.74.3、Vivado 2022
## 使用步骤
1. 环境配置：参考[CFU-Playground](https://github.com/google/CFU-Playground)相关说明
2. 下载本项目代码，放到CFU-Playground项目中的`proj`目录下
3. 连接Arty-A7设备
4. 在本项目所处目录下打开终端，执行以下代码
```sh
source /home/cx/CFU-Playground/env/conda/bin/activate cfu-common && bash
make prog TARGET=digilent_arty USE_VIVADO=1 EXTRA_LITEX_ARGS="--cpu-variant perf+cfu --variant=a7-100 --sys-clk-freq 75000000"
make load BUILD_JOBS=4 TARGET=digilent_arty EXTRA_LITEX_ARGS="--cpu-variant perf+cfu --variant=a7-100 --sys-clk-freq 75000000"
```
5. 执行完成后显示如下界面  
![image](https://github.com/2270142596/WinoFPGA/blob/master/picture/start.png)  
接着在终端界面依次按下`1`、`2`、`1`，即依次选择TfLM Models menu、Mobilenet V1 models、Run test 1
test1对应的图片数据为：
![image](https://github.com/2270142596/WinoFPGA/blob/master/picture/test1.jpg) 
7. 执行结果如下  
![image](https://github.com/2270142596/WinoFPGA/blob/master/picture/acc.png)  
该界面的最上面显示了各个算子在MobilenetV1网络推理过程中的总耗时，单位为千时钟周期。界面最下方probability为图片在该网络模型下十个分类的推理结果，可能性最大的是8，对应物体是船，结果正确。
8. 修改[depthwise_conv.h](https://github.com/2270142596/WinoFPGA/blob/master/src/tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h)中的`const int wino = 1;`为`const int wino = 0;`，将深度卷积加速功能关闭；修改[conv.cc](https://github.com/2270142596/WinoFPGA/blob/master/src/tensorflow/lite/kernels/internal/reference/integer_ops/conv.cc)中的`#ifdef ACCEL_CONV`为`#ifdef NOT_ACCEL_CONV`，将点卷积加速功能关闭。接下来将仅使用RISC-V处理器进行计算。
9. 重新执行步骤4-6，得到如下结果  
![image](https://github.com/2270142596/WinoFPGA/blob/master/picture/notacc.png)  
可以看出，仅使用RISC-V处理器在TensorFlow Lite官方原始代码下的推理结果也为8，与调用加速器的推理结果一致，加速器可以正确计算深度可分离卷积卷积。且没有加速器参与运算，推理耗时明显增加。调用加速器后，MobilenetV1网络的推理时间加快了6.7倍。




