import torch

from model_self import ResNet50

if __name__ == '__main__':
    model = ResNet50()
    input = torch.randn(1, 3, 224, 224)  # B C H W
    output = model(input)