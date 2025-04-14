# ResNet Implementation

这是一个使用PyTorch实现的ResNet-50模型。

## 项目结构
- `model.py`: ResNet50 模型实现
- `data.py`: 数据加载和预处理
- `train.py`: 训练和测试脚本
- `train.bash`: 训练启动脚本

## 环境要求
- Python 3.6+
- PyTorch
- torchvision

## 使用方法
1. 准备数据集，放在`data`目录下
2. 运行训练脚本：
```bash
python train.py