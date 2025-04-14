import torch
from thop import profile
from model import ResNet50

def calc_model_complexity(model):
    input = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, inputs=(input,))

    # 转换GFLOPs
    gflops = flops / 1e9
    # 转换为M参数量
    params_m = params / 1e6

    print(f'GFLOPs: {gflops:.2f}')
    print(f'Params: {params_m:.2f}M')

if __name__ == '__main__':
    model = ResNet50()
    calc_model_complexity(model)