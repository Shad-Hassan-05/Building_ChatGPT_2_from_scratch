from create_net import create_net
from train import train
from load_dataset import load_dataset
from torch import random, save

random.manual_seed(0)
# %%% DO NOT EDIT ABOVE %%%

# Specify the lead_data arguments
data_path = 'iris_dataset.pt'
mean_subtraction = True
normalization = True

# load data set with specific data arguments
iris_dataset = load_dataset(data_path, mean_subtraction, normalization)

# specify the network architecture
in_features = 4 # since there are four attributes (petal len, sepal len, etc.)
out_size = 3
hidden_units = [10, 5 # chose 5, 5 matching the xor and found that 10, 5 leads to better resluts
non_linearity = ['tanH', 'tanH']

# create a network base on the architecture
net = create_net(in_features, hidden_units, non_linearity, out_size)

# specify the training opts
train_opts = {'num_epochs': 80,
              'lr' : 0.01,
              'momentum' : 0.9,
              'batch_size' : 24,
              'weight_decay' : 0.0001,
              'step_size' : 25,
              'gamma' : 1 }

# Train and save the trained model
train(net, iris_dataset, train_opts)
save(net, "iris_solution.pt")
