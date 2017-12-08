### paper
> Pyramid Scene Parsing Network.pdf

### 1. Data - ImageReader.py
- ReaderTrainImageAndLabel 
    - 多线程读取训练的图片和标签
- ReaderTestImage
    - 多线程读取测试的图片

### 2. Model - PSPNet.py
- Network
    - 定义网络的基本操作
- PSPNet
    - 定义网络的具体结构

### 3. Tool - Tools.py
- Tools
    - 工具类

### 4. Runner - RunnerXXX.py
- RunnerTrain.py
    - 训练模型，输入是训练图片和标签的list，输出是训练好的模型
- RunnerOne.py
    - 使用模型，输入是一张图片，输出是语义分割结果
- RunnerAll.py
    - 使用模型，输入是图片的list，输出是语义分割结果


### Reference
* [hellochick/PSPNet-tensorflow](https://github.com/hellochick/PSPNet-tensorflow)
