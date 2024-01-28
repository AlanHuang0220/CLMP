import torch

class Top1Accuracy:
    def __call__(self, output, target):
        """计算 Top-1 准确率"""
        _, predicted = torch.max(output.data, 1)
        correct = (predicted == target).sum().item()
        return correct / output.size(0)