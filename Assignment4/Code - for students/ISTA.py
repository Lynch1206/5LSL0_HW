from matplotlib.image import imread
import numpy as np
from scipy import linalg
import torch

class ISTAI_main(object):
    def __init__(self):
        # ,mu,shrinkage,K,y
        super(ISTAI_main, self).__init__()

    def softthreshold(self,x,theta):
        """softthreshold, x - input, theta - shrinkage parameters: mu*lambda"""      
        return torch.multiply(torch.sign(x) ,torch.maximum(torch.abs(x) - theta, torch.tensor(0.)))

    def ISTA(self,mu,lmbd,K,y:torch.tensor)->torch.tensor:
        """
        # input 
        - y  = input each batch \\
        - mu = constant - step size \\
        - lmbd = Lambda - Lipschitz continuous \\
        - K - max iteration \\
        """        
        # [32,32]
        A = torch.diag_embed(torch.flatten(y)) # Diagnoal matrix - diagonal length = 1024
        x = torch.zeros(len(torch.flatten(y))) # 1024 zero
        error = []

        for i in range(K):
            # g = torch.transpose(A,0,1)@(torch.flatten(y)-A@x) 
            # torch.transpose(A,0,1)
            x_2 = x +mu*A.T@(torch.flatten(y)-A@x) # mu = 1/L
            shrinkage = mu*lmbd
            x_hat = self.softthreshold(x_2, shrinkage)
            error = torch.nn.functional.mse_loss(x_hat,x)
            if error < 1e-7:
                break
            else:
                x = x_hat   

        return torch.tensor(x_hat)

    # def forward(self,mu,lmbd,K,y):
    #     x_hat, error = self.ISTA(mu,lmbd,K,y)
    #     return x_hat, error