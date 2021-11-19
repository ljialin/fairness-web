import torch
import torch.nn.functional as F
import numpy as np
import geatpy as ea

import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
# from load_data import load_data
from torch import nn
# from Evaluate import ana_evaluation
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter


def sigmoid(x):
    return .5 * (1 + np.tanh(.5 * x))
    # return 1.0 / (1.0 + np.exp(-x))
    # if all(x >= 0):
    #     return 1.0 / (1.0 + np.exp(-x))
    # else:
    #     return np.exp(x) / (1.0 + np.exp(x))

#
# class Net_Individual(torch.nn.Module):
#     def __init__(self, n_feature, n_hidden, n_output, dropout=0.5):
#         super(Net_Individual, self).__init__()
#         # self.dropout = torch.nn.Dropout(dropout)
#         self.hidden_1 = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
#         self.relu = nn.ReLU(inplace=True)
#         # self.bn1 = torch.nn.BatchNorm1d(n_hidden)
#
#         self.hidden_2_1 = torch.nn.Linear(n_feature, 64)
#         self.hidden_2_2 = torch.nn.Linear(64, 32)
#         # self.hidden_2 = torch.nn.Linear(n_hidden, n_hidden // 2)
#         # self.bn2 = torch.nn.BatchNorm1d(n_hidden // 2)
#         #
#         # self.hidden_3 = torch.nn.Linear(n_hidden // 2, n_hidden // 4)  # hidden layer
#         # self.bn3 = torch.nn.BatchNorm1d(n_hidden // 4)
#         #
#         # self.hidden_4 = torch.nn.Linear(n_hidden // 4, n_hidden // 8)  # hidden layer
#         # self.bn4 = torch.nn.BatchNorm1d(n_hidden // 8)
#         #
#         # self.out1 = torch.nn.Linear(n_hidden // 8, n_output)  # output layer
#         # self.out1 = torch.nn.Linear(n_hidden, n_output)  # output layer
#         self.out1 = torch.nn.Linear(32, n_output)
#
#     def forward(self, x):
#         # #hidden layers = 1
#         # x = F.relu(self.hidden_1(x))  # activation function for hidden layer
#         # x = self.out1(x)
#
#         # #hidden layers = 2
#         x = F.relu(self.hidden_2_1(x))  # activation function for hidden layer
#         x = F.relu(self.hidden_2_2(x))  # activation function for hidden layer
#         x = self.out2(x)
#
#         # x = self.dropout(self.bn1(x))
#         # x = F.relu(self.hidden_2(x))  # activation function for hidden layer
#         # x = self.dropout(self.bn2(x))
#         # x = F.relu(self.hidden_3(x))  # activation function for hidden layer
#         # x = self.dropout(self.bn3(x))
#         # x = F.relu(self.hidden_4(x))  # activation function for hidden layer
#         # x = self.dropout(self.bn4(x))
#
#         return x


"""
def initialize_parameters_he(layers_dims):
    Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
    ---------------------------------------------------------------
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    ---------------------------------------------------------------
    parameters = {}
    L = len(layers_dims)  # integer representing the number of layers
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2 / layers_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
    return parameters
————————————————
版权声明：本文为CSDN博主「天泽28」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/u012328159/article/details/80025785
"""


def weights_init(m, rand_type='uniformity'):
    # https://github.com/pytorch/examples/blob/master/dcgan/main.py#L114
    # Understanding the difficulty of training deep feedforward neural networks
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        if rand_type == 'uniformity':
            n = m.in_features
            y = 1.0 / np.sqrt(n)
            torch.nn.init.uniform_(m.weight, -y, y)
            # torch.nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            torch.nn.init.zeros_(m.bias)
            # print('weights:', m.weight, '\n bias: ', m.bias)
            # print(' ')
        elif rand_type == 'normal':
            torch.nn.init.normal_(m.weight, 0.0, 0.05)
            torch.nn.init.zeros_(m.bias)


class IndividualNet(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output, dropout=0.3, name='ricci'):
        super(IndividualNet, self).__init__()
        self.name = name
        self.num_hidden = len(n_hidden)
        self.n_hidden = n_hidden
        # if name == 'adult':
        #     if self.num_hidden == 1:
        #         # #hidden layers = 1
        #         self.hidden_1_1 = torch.nn.Linear(n_feature, n_hidden[0])  # hidden layer
        #         # self.out = torch.nn.Linear(n_hidden, n_output)  # output layer
        #         self.out = torch.nn.Linear(n_hidden[0], n_output)
        #     else:
        #         # #hidden layers = 2
        #         self.hidden_2_1 = torch.nn.Linear(n_feature, n_hidden[0])
        #         self.hidden_2_2 = torch.nn.Linear(n_hidden[0], n_hidden[1])
        #
        #         self.out = torch.nn.Linear(n_hidden[1], n_output)
        # elif name == 'ricci':
        #     if self.num_hidden == 1:
        #         # #hidden layers = 1
        #         self.hidden_1_1 = torch.nn.Linear(n_feature, n_hidden[0])  # hidden layer
        #         # self.out = torch.nn.Linear(n_hidden, n_output)  # output layer
        #         self.out = torch.nn.Linear(n_hidden[0], n_output)
        #     else:
        #         # #hidden layers = 2
        #         self.hidden_2_1 = torch.nn.Linear(n_feature, n_hidden[0])
        #         self.hidden_2_2 = torch.nn.Linear(n_hidden[0], n_hidden[1])
        #
        #         self.out = torch.nn.Linear(n_hidden[1], n_output)
        # elif name == 'german':
        #     if self.num_hidden == 1:
        #         # #hidden layers = 1
        #         self.hidden_1_1 = torch.nn.Linear(n_feature, n_hidden[0])  # hidden layer
        #         self.out = torch.nn.Linear(n_hidden[0], n_output)
        #         # self.relu = F.relu
        #     else:
        #         # #hidden layers = 2
        #         self.hidden_2_1 = torch.nn.Linear(n_feature, n_hidden[0])
        #         self.hidden_2_2 = torch.nn.Linear(n_hidden[0], n_hidden[1])
        #
        #         self.out = torch.nn.Linear(n_hidden[1], n_output)
        # elif name == 'propublica-recidivism':
        #     if self.num_hidden == 1:
        #         # #hidden layers = 1
        #         self.hidden_1_1 = torch.nn.Linear(n_feature, n_hidden[0])  # hidden layer
        #         # self.out = torch.nn.Linear(n_hidden, n_output)  # output layer
        #         self.out = torch.nn.Linear(n_hidden[0], n_output)
        #     else:
        #         # #hidden layers = 2
        #         self.hidden_2_1 = torch.nn.Linear(n_feature, n_hidden[0])
        #         self.hidden_2_2 = torch.nn.Linear(n_hidden[0], n_hidden[1])
        #         self.out = torch.nn.Linear(n_hidden[1], n_output)

        if self.num_hidden == 1:
            # #hidden layers = 1
            self.hidden_1_1 = torch.nn.Linear(n_feature, n_hidden[0])  # hidden layer
            # self.out = torch.nn.Linear(n_hidden, n_output)  # output layer
            self.out = torch.nn.Linear(n_hidden[0], n_output)
        else:
            # #hidden layers = 2
            self.hidden_2_1 = torch.nn.Linear(n_feature, n_hidden[0])
            self.hidden_2_2 = torch.nn.Linear(n_hidden[0], n_hidden[1])

            self.out = torch.nn.Linear(n_hidden[1], n_output)
        # else:
        # self.hidden1 = torch.nn.Linear(n_feature, hidden_units)
        # self.hidden2 = torch.nn.Linear(hidden_units, hidden_units)
        # self.hidden3 = torch.nn.Linear(hidden_units, n_output)
        self.dropout_value = dropout
        if dropout > 0:
            self.dropout = nn.Dropout(self.dropout_value)
        else:
            self.dropout = None
        self.relu = nn.ReLU()

    def forward(self, x):
        #hidden layers = 2
        # if self.name == 'adult':
        #     if self.num_hidden == 1:
        #         x = self.hidden_1_1(x)
        #         if self.dropout_value > 0:
        #             x = self.dropout(x)
        #
        #         x = self.relu(x)
        #         pred_logits = self.out(x)
        #     else:
        #         x = self.hidden_2_1(x)
        #         if self.dropout_value > 0:
        #             x = self.dropout(x)
        #         x = self.relu(x)
        #         x = self.hidden_2_2(x)
        #         if self.dropout_value > 0:
        #             x = self.dropout(x)
        #         x = self.relu(x)
        #         pred_logits = self.out(x)
        # elif self.name == 'ricci':
        #     if self.num_hidden == 1:
        #         x = F.relu(self.hidden_1_1(x))  # activation function for hidden layer
        #         pred_logits = self.out(x)
        #     else:
        #         x = F.relu(self.hidden_2_1(x))  # activation function for hidden layer
        #         x = F.relu(self.hidden_2_2(x))  # activation function for hidden layer
        #         pred_logits = self.out(x)
        # elif self.name == 'german':
        #     if self.num_hidden == 1:
        #         x = F.relu(self.hidden_1_1(x))  # activation function for hidden layer
        #         pred_logits = self.out(x)
        #     else:
        #         x = F.relu(self.hidden_2_1(x))  # activation function for hidden layer
        #         x = F.relu(self.hidden_2_2(x))  # activation function for hidden layer
        #         pred_logits = self.out(x)
        # elif self.name == 'propublica-recidivism':
        #     if self.num_hidden == 1:
        #         x = F.relu(self.hidden_1_1(x))  # activation function for hidden layer
        #         if self.dropout > 0:
        #             x = F.dropout(x, p=self.dropout)
        #         pred_logits = self.out(x)
        #     else:
        #         x = F.relu(self.hidden_2_1(x))  # activation function for hidden layer
        #         x = F.relu(self.hidden_2_2(x))  # activation function for hidden layer
        #         pred_logits = self.out(x)
        # else:
        #     if self.num_hidden == 1:
        #         x = F.relu(self.hidden_1_1(x))  # activation function for hidden layer
        #         x = F.dropout(x, p=self.dropout)
        #         pred_logits = self.out(x)
        #     else:
        #         x = F.relu(self.hidden_2_1(x))  # activation function for hidden layer
        #         x = F.relu(self.hidden_2_2(x))  # activation function for hidden layer
        #         pred_logits = self.out(x)

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

        pred_label = torch.sigmoid(pred_logits)
        # # hidden = 1
        # x = F.relu(self.hidden1(x))  # activation function for hidden layer
        # x = F.dropout(x, p=self.dropout)
        # pred_logits = self.hidden2(x)  # activation function for hidden layer
        # pred_label = torch.sigmoid(pred_logits)

        # x = F.relu(self.hidden1(x))  # activation function for hidden layer
        # # x = F.dropout(x, p=self.dropout)
        # x = F.relu(self.hidden2(x))
        # pred_logits = self.hidden3(x)  # activation function for hidden layer
        # pred_label = torch.sigmoid(pred_logits)
        return pred_logits, pred_label


def mutate(model, var):

    with torch.no_grad():
        for name, param in model.named_parameters():
            weighs = np.array(param.detach())
            # print(name)
            # print('  before: ', weighs)
            weighs += np.random.normal(loc=0, scale=var, size=param.shape)
            model.state_dict()[name].data.copy_(torch.Tensor(weighs))
            # print('  after weights: ', weighs)
            # print('  after data: ', model.state_dict()[name].data)

    # with torch.no_grad():
    #     for name, param in model.named_parameters():
    #         print(name)
    #         print(param)
    return model


class Population_NN:

    def __init__(self, train_data_norm, train_data, train_y, test_data, test_data_norm, test_y, pop_size, n_feature,
                 n_hidden, n_output, sensitive_attributions, positive_y):
        self.train_data = train_data
        self.test_data = test_data
        self.train_y = train_y
        self.test_y = test_y
        self.pop_size = pop_size
        self.n_feature = n_feature
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.population = None
        self.learning_rate = 0.01
        self.batch_size = 500
        self.positive_y = positive_y
        self.train_data_norm = train_data_norm
        self.test_data_norm = test_data_norm
        self.netpara = None
        self.sensitive_attributions = sensitive_attributions

    def initialization(self):
        population = []
        for i in range(self.pop_size):
            pop = IndividualNet(self.n_feature, self.n_hidden, self.n_output)
            pop.apply(weights_init)
            population.append(pop)
        self.population = population

    def train_model(self):
        All_logits = np.array([])
        PopObj = np.zeros([self.pop_size, 3])
        Groups_info = []
        for idx in range(self.pop_size):
            individual = self.population[idx]
            x_train = torch.Tensor(self.train_data_norm)
            y_train = torch.Tensor(self.train_y)
            y_train = y_train.view(y_train.shape[0], 1)

            x_test = torch.Tensor(self.test_data_norm)
            y_test = torch.Tensor(self.test_y)
            y_test = y_test.view(y_test.shape[0], 1)

            optimizer = torch.optim.Adam(individual.parameters(), lr=self.learning_rate, weight_decay=1e-5)
            loss_fn = torch.nn.BCEWithLogitsLoss()  # Combined with the sigmoid

            train = TensorDataset(x_train, y_train)
            train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
            individual.train()
            avg_loss = 0.

            for i, (x_batch, y_batch) in enumerate(train_loader):
                y_pred = individual(x_batch)

                loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()  # clear gradients for next train
                loss.backward()  # -> accumulates the gradient (by addition) for each parameter
                optimizer.step()  # -> update weights and biases
                avg_loss += loss.item() / len(train_loader)
            with torch.no_grad():
                a = 0
                if a == 1:
                    # print('The formation in test data: ')
                    logits = sigmoid(np.array(individual(x_test).detach()))
                    accuracy, individual_fairness, group_fairness, Groups = ea.ana_evaluation(self.test_data,
                                                                                           self.test_data_norm,
                                                                                           logits, y_test,
                                                                                           self.sensitive_attributions,
                                                                                           2)
                else:
                    # print('The formation in train data: ')
                    logits = sigmoid(np.array(individual(x_train).detach()))
                    accuracy, individual_fairness, group_fairness, Groups = ea.ana_evaluation(self.train_data,
                                                                                           self.train_data_norm,
                                                                                           logits, y_train,
                                                                                           self.sensitive_attributions,
                                                                                           2)
                # print('  accuracy: %.4f, individual fairness: %.5f, group fairness: %.5f\n'
                #       % (accuracy, individual_fairness, group_fairness))

                Groups_info.append(Groups)
                PopObj[idx][:] = np.array([accuracy, individual_fairness, group_fairness])
                if idx != 0:
                    All_logits = np.concatenate([All_logits, logits], axis=1)
                else:
                    All_logits = np.array(logits)

        return PopObj, Groups_info

    def mutation(self, idx):
        Offspring = []
        for pop_idx in idx:
            parent = self.population[pop_idx]
            parent = mutate(parent, 0.001)
            Offspring.append(parent)

        return Offspring


# [train_data, train_data_norm, test_data, test_data_norm, train_label, test_label, train_y, test_y,
#  positive_y] = load_data('>50K')
# print('Data has been already loaded')
#
# pop = Population_NN(train_data_norm=train_data_norm, train_data=train_data, train_y=train_y, test_data=test_data,
#                  test_data_norm=test_data_norm, test_y=test_y, pop_size=5, n_feature=train_data.shape[1],
#                  n_hidden=100, n_output=1, positive_y=positive_y, sensitive_attributions=['gender', 'race'])
# pop.initialization()
# PopObj, Groups_info = pop.train_model()
# print(PopObj)
#
# good_one = np.array([0, 1, 2, 3, 4])
# # a = mutate(pop.population[0], 0.01)
# off = pop.mutation(good_one)

