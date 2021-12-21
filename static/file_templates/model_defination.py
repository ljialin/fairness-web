"""
  @Time : 2021/9/4 14:05 
  @Author : Ziqi Wang
  @File : mlp.py 
"""

from torch import nn


class MLP(nn.Module):
    def __init__(self, num_in, num_hiddens):
        super(MLP, self).__init__()
        num_neurons = [num_in, *num_hiddens]
        layers = []
        for i in range(1, len(num_neurons)):
            layers.append(nn.Linear(num_neurons[i - 1], num_neurons[i]))
            layers.append(nn.ReLU())
        self.main = nn.Sequential(
            *layers, nn.Linear(num_neurons[-1], 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

def main():
    return MLP(60, [64])
