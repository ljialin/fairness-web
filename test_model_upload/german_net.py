from torch import nn, sigmoid


class IndividualNet(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output, dropout=0.3, name='ricci'):
        super(IndividualNet, self).__init__()
        self.name = name
        self.num_hidden = len(n_hidden)
        self.n_hidden = n_hidden
        if self.num_hidden == 1:
            # #hidden layers = 1
            self.hidden_1_1 = nn.Linear(n_feature, n_hidden[0])  # hidden layer
            # self.out = torch.nn.Linear(n_hidden, n_output)  # output layer
            self.out = nn.Linear(n_hidden[0], n_output)
        else:
            # #hidden layers = 2
            self.hidden_2_1 = nn.Linear(n_feature, n_hidden[0])
            self.hidden_2_2 = nn.Linear(n_hidden[0], n_hidden[1])

            self.out = nn.Linear(n_hidden[1], n_output)
        self.dropout_value = dropout
        if dropout > 0:
            self.dropout = nn.Dropout(self.dropout_value)
        else:
            self.dropout = None
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.num_hidden == 1:
            x = self.hidden_1_1(x)
            if self.dropout_value > 0:
                x = self.dropout(x)

            x = self.relu(x)
            pred_logits = self.out(x)
        else:
            x = self.hidden_2_1(x)
            if self.dropout_value > 0:
                x = self.dropout(x)
            x = self.relu(x)
            x = self.hidden_2_2(x)
            if self.dropout_value > 0:
                x = self.dropout(x)
            x = self.relu(x)
            pred_logits = self.out(x)

        pred_label = sigmoid(pred_logits)
        return pred_logits, pred_label

def main():
    return IndividualNet(56, [64], 1)
