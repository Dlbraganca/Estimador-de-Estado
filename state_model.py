import torch
import torch.nn as nn
import numpy as np
from src.import_data import read_med, read_system
from estimador import StateEstimator

class NeuralNetwork(nn.Module):
    def __init__(self, n_input_features, n_hidden_size,  n_output_features):
        super(NeuralNetwork, self).__init__()
        #first layer
        self.linear1 = nn.Linear(n_input_features, n_output_features)
        #self.relu1 = sigmoid(x)
        #second layer
        # self.linear2 = nn.Linear(int(n_hidden_size/2), int(n_hidden_size/4))
        # # self.relu2 = nn.Sigmoid()
        # #third
        # self.linear3 = nn.Linear(int(n_hidden_size/4), n_output_features)
        # self.relu3 = nn.ReLU()

    def forward(self, x):
        #first layer
        y_predicted = self.linear1(x)
        #y_predicted = self.relu1(y_predicted)
        # #second layer
        # y_predicted = self.linear2(y_predicted)
        # # y_predicted = self.relu2(y_predicted)
        # #third layer
        # y_predicted = self.linear3(y_predicted)
        # # y_predicted = self.relu3(y_predicted)
        return y_predicted

med = np.array(read_med(), dtype=np.float32)
sys = np.array(read_system(), dtype=np.float32)

model = None

for i in range(200):
    atual = i
    next = i + 1
    ee = StateEstimator(sys=sys, med=med[next])
    ee.Begin()
    state = ee.get_state()
    nBus = ee.get_bus_num()
    v_state = state[nBus-1:]
    teta_state = state[0:nBus-1]

    x = torch.tensor(med[atual,:,5], requires_grad=True)
    #y = torch.from_numpy(med[i+1,:,5])
    y = torch.tensor([teta_state[0].item(), v_state[0].item()])

    inputs = int(np.shape(x)[0])
    outputs = int(np.shape(y)[0])

    model = NeuralNetwork(inputs, outputs, outputs)

    #2) loss and optimizer 
    learning_rate = 1
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_final = 0
    
    #3) learning loop
    num_epochs = 100
    for epoch in range(num_epochs):
        #foward pass and loss calculation
        y_predicted = model(x)
        loss = criterion(y_predicted, y)
        loss_final = loss
        
        #backward pass
        loss.backward()
        
        #updates
        optimizer.step()

        #zero gradient
        optimizer.zero_grad()
    
#teste

atual = 0
next = 0 + 1
ee = StateEstimator(sys=sys, med=med[next])
ee.Begin()
state = ee.get_state()
nBus = ee.get_bus_num()
v_state_next = state[nBus-1:]
teta_state_next = state[0:nBus-1]
x = torch.tensor(med[atual,:,5], requires_grad=True)
y = model(x)
#previsao
print('previsão')
print(y)
#real
print('real')
print(teta_state_next[0].item())
print(v_state_next[0].item())
