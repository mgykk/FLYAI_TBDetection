# FLYAI_TBDetection

## 比赛简介

结核病（Tuberculosis，TB)是由结核分枝杆菌（Mycobacterium tuberculosis) 引起的一种慢性人畜共患病，它不受年龄、性别、种族、职业、地区的影响，人体许多器官、系统均可患结核病，其中以肺结核最为常见。结核病既是一个公共卫生问题，也是一个社会经济问题，对人类的公共健康构成很大威胁，因此对其快速诊断检测就至关重要。

染色处理可以使得结核杆菌在显微镜拍摄的医学图像中显现，医生则可以通过检测图像中的结核杆菌辅助诊断患者是否有结核病。

通过构建准确率的目标检测模型可实现由智能系统辅助医生进行检测工作，应用于目前的医疗检测产品中能够满足真实的结核病检测需求。

图像数据：

![](https://github.com/mgykk/FLYAI_TBDetection/blob/master/data/107.jpg)

## 数据处理方法

- 调整图像及检测框的大小
- 适度旋转图像以及对应标签
- 随机添加高斯噪声
- 由于医学图像特性，在尝试添加图像的亮度，对比度，饱和度浮动后导致精度下降

以上方法均添加概率使其成为概率事件。

## 模型选择及超参数设置

由于比赛刚刚开始，因此当前测试的模型是Faster-RCNN-ResNet50-FPN，直接由torchvision获得，修改预测头类别数进行训练。

后续有时间的话可能会尝试一下EfficientDet，YoloV4，YoloV5（emmm，不知道能不能成功~）

超参数设置部分：

Lr=0.001，Epoch=10，Batch_size=1

优化器选择：

采用SGD(随机梯度下降法)对网络中的参数进行迭代更新

## 训练结果

由于比赛刚刚开放，因此此排名仅代表当前排名

![]()
