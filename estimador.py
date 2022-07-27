import numpy as np
from src.jacobiano import Jacob
from src.y_barra import Y_bus
from src.matriz_covariancia import Cov_m
from src.flat_start import Flat_start
from src.h_vetor import h
from src.import_data import read_med
from src.import_data import read_system
from src.num_bus import get_num_bus
from src.analise_residual import residual_analisys

class StateEstimator():


    def __init__(self,  path_sys = None, 
                        path_med = None, 
                        tolerance = 0.001,
                        sys = np.array([0]),
                        med = np.array([0])):

        #INICIALIZAÇÃO PADRÃO (USA UM CASO EXEMPLO)
        if (med.max() == 0 or sys.max() == 0)and(path_sys == None or path_med == None):
        #SISTEMA
        #[DE,PARA,IMPENDACIA,REATANCIA. CONDUTÂNCIA SHUNT,SUSCEPTÂNCIA SHUNT]
            # sys = [ [1,2,0.00,0.40,0, 0],
            #         [1,3,0.00,0.20,0, 0],
            #         [2,3,0.00,0.25,0, 0],
            #         [2,4,0.00,0.50,0, 0],
            #         [3,4,0.00,0.25,0, 0]]   
            sys =  [[1,2,0.01,0.03,0,0],
                    [1,3,0.02,0.05,0,0],
                    [2,3,0.03,0.08,0,0]]

        #MEDIÇÕES
        #[NÚMERO, TIPO, DE, PARA, VALOR REAL, VALOR MEDIDO, VARIANCIA]
        #1-FLUXO DE POTÊNCIA ATIVA
        #2-FLUXO DE POTÊNCIA REATIVA
        #3-INJEÇÃO DE POTÊNCIA ATIVA
        #4-INJEÇÃO DE POTêNCIA REATIVA
        #5-MÓDULO DE TENSÃO

            # med = [ [1,1,1,2,0,0.70,0.0011],
            #         [2,2,1,2,0,0.15,0.0011],
            #         [3,1,1,3,0,-0.35,0.0011],
            #         [4,2,1,3,0,-0.10,0.0011],
            #         [5,3,2,0,0,1.40,0.0011],
            #         [6,4,2,0,0,0.85,0.0011],
            #         [7,3,4,0,0,-0.30,0.0011],
            #         [8,4,4,0,0,-0.25,0.0011],
            #         [9,5,1,0,0,1.02,0.000011],
            #         [10,5,3,0,0,0.99,0.000011]
            #         ]
            med =   [[1,1,1,2,0,+0.888,0.000064],
                    [2,1,1,3,0,+1.173,0.000064],
                    [3,3,2,0,0,-0.501,0.0001],
                    [4,2,1,2,0,+0.568,0.000064],
                    [5,2,1,3,0,+0.663,0.000064],
                    [6,4,2,0,0,-0.286,0.0001],
                    [7,5,1,0,0,+1.006,0.000016],
                    [8,5,2,0,0,+0.968,0.000016]]

            self.sys = np.array(sys)
            self.med = np.array(med)        
            self.tolerance = tolerance
            self.nBus = get_num_bus(self.sys)
            self.yBus = Y_bus(self.sys)
            self.r = Cov_m(self.med)
            self.x = Flat_start(self.nBus)
            self.z = np.matrix(self.med[:,5])
            self.z_real = np.matrix(self.med[:,4])
        #INCIALIZAÇÃO NORMAL(A PARTIR DO CASO)
        
        else:

            #INICIALIZAÇÃO COM MEDIDAS
            if(path_sys == None or path_med == None):
                self.med = med
                self.sys = sys

            #INICIALIZAÇÃO COM ARQUIVOS DE TEXTO
            else:

                self.sys = read_system(path=path_sys)
                self.med = read_med(path=path_med)

            self.tolerance = tolerance
            self.nBus = get_num_bus(self.sys)
            self.yBus = Y_bus(self.sys)
            self.r = Cov_m(self.med) 
            self.x = Flat_start(self.nBus)
            self.z = np.matrix(self.med[:,5])
            self.z_real = np.matrix(self.med[:,4])

    def Begin(self):
        #DEFININE PARAMETROS DE PRECISAO
        #np.set_printoptions(precision=2, suppress=True)

        sys = self.sys
        med = self.med
        tolerance = self.tolerance
        y_bus = self.yBus
        z = self.z.transpose()

        nbus = self.nBus

        R = self.r
        Rinv = np.linalg.inv(R)
        #np.savetxt('Rtext.txt',Rinv,fmt='%.2f')

        x = Flat_start(nbus)
        
        countIt = 0

        while (True):
            countIt = countIt + 1
            
            H = Jacob(x, sys, y_bus, med, nbus)
            
            Htrans = H.transpose()
            hx = h(x, sys, med, nbus, y_bus)
            
            G = Htrans*Rinv*H
            #TESTE DE OBSERVABILIDADE
            #try:
            #np.savetxt('text.txt',G,fmt='%.2f')
            Ginv = np.linalg.inv(G)
            #except:
            #    print("SISTEMA NÃO OBSERVÁVEL")
            #    return 1
            #else:
            fx = H.transpose()*Rinv*(z - hx)
            # fx = fx*(z - hx)
            dx = Ginv*fx
            #dx = dx[:,0]
            if np.max(np.absolute(dx)) <= tolerance:
                x = x + dx
                self.x = x
                self.h = h(self.x, self.sys, self.med, self.nBus, self.yBus).transpose()
                self.H = H
                self.G = G
                self.iterations = countIt
                return 0
            else:
                x = x + dx

        
    def get_med(self):
        return self.z

    def get_state(self):
        return self.x

    def get_iteration_count(self):
        return self.iterations
    
    def get_y_bus(self):
        return self.yBus

    def get_vector_size(self):
        return self.x.shape[0]

    def get_bus_num(self):
        return self.nBus

    def get_z(self):
        return self.z

    def get_residual(self):
        H = Jacob(self.x, self.sys, self.yBus, self.med, self.nBus)
        return residual_analisys(med=self.med, estimated_med=self.h, H=H, R=self.r)
    
    def get_h(self):
        return self.h
    