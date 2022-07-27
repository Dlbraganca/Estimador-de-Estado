from importlib.resources import path
from estimador import StateEstimator
from src.import_data import read_med, read_system
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np


sys = read_system('data/system.txt')
med = read_med('data/med.txt')


#estado = StateEstimator(sys = sys, med = med[0])
#estado = StateEstimator(sys = sys, med = med, tolerance=0.001)
#estado = StateEstimator(path_sys='data/14_bus.txt',path_med ='data/med_14bus copy.txt', tolerance=0.0005)
estado = StateEstimator()
#print(estado.get_y_bus())
estado.Begin()
print(estado.get_med())
print(estado.get_state())
print(estado.get_iteration_count())
print(torch.rand(5,3))
print(torch.cuda.is_available())
