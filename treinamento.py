import torch
import torch.nn as nn
import numpy as np
from src.import_data import read_med, read_system
from estimador import StateEstimator
from random import randint
import matplotlib.pyplot as plt

def get_x(var):
    x = []
    for i in var:
        x.append(i.item())
    return x

def plot(x_axis_real = [], x_axis_model= [], x_axis_test = [], y_axis_real = [], y_axis_model = [], y_axis_test = [], num_med = 0):
    plt.plot(x_axis_model, y_axis_model, label='Previsão')
    plt.plot(x_axis_real, y_axis_real, ".r", label='Real')
    plt.plot(x_axis_test, y_axis_test, ".g", label='Teste')
    #plt.plot(x_plot, y_before, ".g", label='Medido')
    plt.xlabel("Instantes de Tempo")
    plt.ylabel("Valor da Medição (pu)")
    plt.title("Modelo de Previsão para a Medida " + str(num_med) )
    plt.legend()
    plt.show()

class NeuralNetwork(nn.Module):
    def __init__(self, n_input_features, n_hidden_size,  n_output_features):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(n_input_features, n_hidden_size, dtype=torch.float64)
        self.activation_function1 = nn.Identity()
        self.linear2 = nn.Linear(n_hidden_size, n_output_features, dtype=torch.float64)
        self.activation_function2 = nn.Identity()

    def forward(self, x):
        y_predicted = self.linear1(self.activation_function1(x))
        y_predicted = self.linear2(self.activation_function2(y_predicted))

        
        return y_predicted

device = torch.device('cuda')
#device = 'cpu' 

med = np.array(read_med(), dtype=np.float64)
sys = np.array(read_system(), dtype=np.float64)

size_meds = int(np.shape(med)[0])

num_med = 23 #define o número de entradas para o modelo

trained_measure = 2 # definindo a medida a ser treinada

for measure in range(trained_measure - 1, trained_measure):   

    PATH = './models_0/+' + str(measure+1) + '.pth'

    model = NeuralNetwork(num_med, num_med, 1).to(device)
    x = []
    y = []
    x_test = []
    y_test = []
    x_axis = []
    x_axis_training = []
    x_axis_test = []

    #gerar numeros randomicos (separando teste de treinamento)
    qnt_test = 15 #quantidade de valores teste
    qnt_values = 256 #valor total de dados
    test_set = []
    while(len(test_set) < qnt_test):
        value = randint(0,qnt_values - 1)
        if(not(value in test_set)):
            test_set.append(value)
    #
    #executa o estimador de estado para todos os conjuntos de medidas
    #aqui são obtidos os dados de entrada do sistema
    for i in range(np.shape(med)[0] - 1):
        now = i
        next = now + 1
        ee = StateEstimator(sys=sys, med=med[now])
        ee.Begin()

        state = ee.get_state()
        nBus = ee.get_bus_num()

        #separa as os dados de treinamento dos de teste
        if (i in test_set):
        #dados de teste
            x_axis_test.append(next)
            x_test.append(get_x(state)[0:23])
            y_test.append([med[next, measure, 4].item()])
        #dados de treinamento
        else:    
            x.append(get_x(state)[0:23])
            x_axis.append(next)
            y.append([med[next, measure, 4].item()])
    #

    #2) loss and optimizer 
    learning_rate = 0.0001
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_final = 0


    #3) learning loop
    num_epochs = 100
    for epoch in range(num_epochs):
        #foward pass and loss calculation
        x_torch = torch.tensor(x, dtype=torch.float64, requires_grad=True, device=device)
        #x_torch = nn.functional.normalize(x_torch, dim=1)
        y_torch = torch.tensor(y, dtype=torch.float64, requires_grad=True, device=device)
        y_predicted = model(x_torch)
        loss = criterion(y_predicted, y_torch)
        #print(loss.item())
        if loss.item() <= 0.0001:
            break
        
        #backward pass
        loss.backward()
        
        #updates
        optimizer.step()

        #zero gradient
        optimizer.zero_grad()
        #print dos pesos
        # for param in model.parameters():
        #     print(param.data)


    y_predicted_vector = []
    y_before = []
    y_real = []
    x_real = []

    #modelulo de plotagem dos resultados
    for i in range(len(x) - 1):
        atual = i
        next = atual + 1
        x_result = torch.tensor(x[i], dtype=torch.float64, device=device)
        j = x_axis[i] #pega a a proxima medida
        y_predicted_vector.append(model(x_result).item())
        y_before.append(med[j,measure,5].item())
        y_real.append(med[j,measure,4].item())
        x_real.append(j)


    y_predicted = np.array(y_predicted_vector)
    y_before = np.array(y_before)
    y_real = np.array(y_real)

    # print('medida treinada: ', measure + 1)
    # print('medida prevista', y_predicted)
    # print('medida original', y_real)
    # print('medida anterior', y_before)
    # print('medida: ', measure+1)
    
    # print('diferenca entre original e previsto: ', np.linalg.norm(y_real-y_predicted))

    # print('diferença real: ', np.linalg.norm(y_before-y_real))

    torch.save(model, PATH)

    plot(x_axis_model = x_real, y_axis_model = y_predicted_vector,
        x_axis_test = x_axis_test, y_axis_test = y_test,
        x_axis_real= x_real,y_axis_real =  y_real, num_med=trained_measure)

    del model