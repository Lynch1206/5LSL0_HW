import torch
from torch import nn
from torch.autograd import Variable
class smoother(nn.Module):
    def __init__(self):
        # ,mu,shrinkage,K,y
        super(smoother,self).__init__()
        self.weight = Variable(torch.ones(1),requires_grad=True)
        
    def forward(self,x):
        """smooth forwarder
        return x + 0.5*(torch.sqrt( torch.square(x + self.weight) + 1) - torch.sqrt( torch.square(x + self.weight) + 1) )
        """

        x = x + 0.5*(torch.sqrt( torch.square(x + self.weight) + 1) - torch.sqrt( torch.square(x + self.weight) + 1) )
        return x



class LISTA_(nn.Module):
    def __init__(self,k=3):
        super(LISTA_,self).__init__()
        """ 2 convolutional layers & 1 smoother layer """
        self.x = torch.nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,padding=1) 
        self.y = torch.nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,padding=1) 
        self.k = k
        self.smoother = nn.ModuleList(smoother() for i in range(self.k))
        

    def forward(self,y):
        x = torch.zeros_like(y,requires_grad=False)
        for i in range(self.k):
            y_ = self.y(y)
            x_ = self.x(x)
            x = self.smoother[i](y_+x_)

        return x

