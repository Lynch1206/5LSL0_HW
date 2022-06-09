from matplotlib.image import imread
import numpy as np
from scipy import linalg
import torch

class ISTAI_main(object):
    def __init__(self):
        # ,mu,shrinkage,K,y
        super(ISTAI_main, self).__init__()
        # self.mu = mu                # Step size
        # self.shrinkage = shrinkage  # shrinkage parameters 
        # self.K = K                  # Interation
        # self.y=y

    
    def softthreshold(self,x,shrinkage):        
        # torch.sign(x) Returns a new tensor with the signs of the elements of input.
        return torch.multiply(torch.sign(x) ,torch.maximum(torch.abs(x) - shrinkage, torch.tensor(0.)))



    def ISTA(self,mu,lmbd,K,y:torch.tensor)->torch.tensor:
        """
        y = A  # input each batch
        """        
        # 32,32
        A = torch.diag_embed(torch.flatten(y)) # Diagnoal matrix - diagonal length = 1024
        x = torch.zeros(len(torch.flatten(y))) # 1024 zero
        error = []
        for i in range(K):
            g = A.T@(torch.flatten(y)-A@x) 
            x_2 = x +mu*g
            x_hat = self.softthreshold(x_2, mu*lmbd)            
            error = torch.nn.functional.mse_loss(x,x_hat)
            if error < 1e-15:
                break
            else:
                x = x_hat

        return torch.tensor(x_hat)

    # def forward(self,mu,lmbd,K,y):
    #     x_hat, error = self.ISTA(mu,lmbd,K,y)
    #     return x_hat, error