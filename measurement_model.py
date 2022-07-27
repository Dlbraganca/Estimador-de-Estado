import torch
import torch.nn as nn
import numpy as np
from src.import_data import read_med, read_system
from estimador import StateEstimator

def sigmoid(x):
    return 1./(1+torch.exp(-x/5))

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

size_meds = int(np.shape(med)[0])
num_med = int(np.shape(med)[1])
print(num_med)
    
measure = 0 #numero da medida para ser treinada

model = NeuralNetwork(num_med, num_med, 1)

for i in range(size_meds - 3):
    
    x = torch.tensor(med[i,:,5], requires_grad=True)
    #y = torch.from_numpy(med[i+1,:,5])
    y = torch.tensor([med[i+1,measure,5].item()])
    
    #2) loss and optimizer 
    learning_rate = 0.0001
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_final = 0
    
    #3) learning loop
    num_epochs = 10000000 
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

print('ulimo gradiente: ', loss_final.item())

x = torch.from_numpy(med[0,:,5])
y = torch.tensor(med[1,measure,5])
y_predicted_vector = []
y_before = []
y_after = []
y_predicted = model(x)
for i in range(np.shape(med)[0] - 1):
    
    x = torch.from_numpy(med[i,:,5])
    y = torch.tensor(med[i+1,measure,5])

    y_predicted_vector.append(model(x).item())
    y_before.append(med[i,measure,5].item())
    y_after.append(med[i+1,measure,5].item())

y_predicted = np.array(y_predicted_vector)
y_before = np.array(y_before)
y_after = np.array(y_after)

print('medida treinada: ', measure + 1)
print('medida prevista', y_predicted)
print('medida original', y_after)
print('diferenca entre original e previsto: ', np.linalg.norm(y_after-y_predicted))
print('')

print('diferença real: ', np.linalg.norm(y_before-y_after))
print(' ')