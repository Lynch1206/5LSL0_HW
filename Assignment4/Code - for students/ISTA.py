from matplotlib.image import imread
import numpy as np
from scipy import linalg
import torch

class ISTAI_main(object):
    def __init__(self,mu,shrinkage,K,y):
        # super(ISTAI_main, self).__init__()
        self.mu = mu                # Step size
        self.shrinkage = shrinkage  # shrinkage parameters 
        self.K = K                  # Interation
        self.y=y

    
    def softthreshold(self,x,shrinkage):        
        # torch.sign(x) Returns a new tensor with the signs of the elements of input.
        return (torch.sign(x) * torch.maximum(torch.abs(x) - torch.tensor(shrinkage), torch.tensor(0.)))

    def ISTA(self,mu,lmbd,K,y:torch.tensor)->torch.tensor:
        """
        y = A  # input each batch
        """        
        # x = torch.flatten(y)

        A = torch.diag_embed(torch.flatten(y)) # Diagnoal matrix 
        x = torch.zeros(len(torch.flatten(y))) # 1024 zero
        error = []

        for i in range(K):

            g = A.T@(torch.flatten(y)-A@x)
            x_2 = x +mu*g
            x_hat = self.softthreshold(x_2, mu*lmbd)
            
            error = torch.nn.functional.mse_loss(x_hat,x)
            if error < 1e-9:
                break
            else:
                x = x_hat
        recon = x_hat.reshape(1,32,32)

        return recon, x_hat, error,i