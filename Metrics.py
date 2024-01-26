import torch

def top1_accuracy(output, target):
    """計算 Top-1 準確率"""
    _, predicted = torch.max(output.data, 1)
    print(predicted)
    print(target)
    correct = (predicted == target).sum().item()
    print(correct)
    return correct / output.size(0)