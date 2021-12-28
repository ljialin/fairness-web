import torch

class IndividualNet(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output=1, dropout=0.3):
        super(IndividualNet, self).__init__()

        num_neurons = [n_feature, *n_hidden]
        self.main = torch.nn.Sequential()
        for i in range(1, len(num_neurons)):
            self.main.add_module("linear_{}".format(str(i)), torch.nn.Linear(num_neurons[i - 1], num_neurons[i]))
            self.main.add_module("dropout_{}".format(str(i)), torch.nn.Dropout(dropout))
            self.main.add_module("relu_{}".format(str(i)), torch.nn.ReLU())
        self.main.add_module("out", torch.nn.Linear(num_neurons[-1], n_output))
        self.main.add_module("sigmoid", torch.nn.Sigmoid())

    def forward(self, x):
        return self.main(x)

def get_model():
    return IndividualNet(60, [128])