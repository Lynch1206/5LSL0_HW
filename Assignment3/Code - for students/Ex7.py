# %% imports
import torch
import torch.nn as nn
# Build an Encoder for classfication
# %%  Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, out_channels = 16, kernel_size=3, padding=1), # Padding = 1 ensures that the final output is the same size as input 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                      # Downsampling to 16x16x16

            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                      # Downsampling to 8x8x16

            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                      # Downsampling to 4x4x16

            # nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size=3, padding=1), 
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(2),                      # Downsampling to 2x2x16

            nn.Flatten()                   # Flatten 2x2x16 to 1-dim
        )
        self.Emu = nn.Linear(2*2*16,2)
        self.E_logVar= nn.Linear(2*2*16,2)
        self.N = torch.distributions.Normal(0,1) # normal dis sample
        self.KL = 0 # not KL divergence yet
        
    def forward(self, x):
        # use the created layers here
        X =  self.encoder(x)
        mu = self.Emu(x)
        LogVar = self.E_logVar(x)
        Std = torch.exp(LogVar/2)
        LaVar = mu+Std*self.N.sample(mu.shape)
        # calculate KL
        self.KL = (Std**2 + mu**2 - torch.log(Std)-0.5).sum()
        return mu, Std, LaVar
# %%  Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.DE = nn.Sequential(
            nn.Linear(2,16),
            nn.ReLU(inplace=True)
        )
        # create layers here
        self.Decoder = nn.Sequential(
            nn.ConvTranspose2d(10 , out_channels = 16, kernel_size=3, padding=1), # Padding = 1 ensures that the final output is the same size as input 
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(2,1), mode='bilinear'),                     # Upsampling to 2x2x16
            nn.ConvTranspose2d(in_channels = 16 , out_channels = 16, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'),                     # Upsampling to 4x4x16
            nn.ConvTranspose2d(in_channels = 16 , out_channels = 16, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'),                     # Upsampling to 8x8x16
            nn.ConvTranspose2d(in_channels = 16 , out_channels = 16, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'),                     # Upsampling to 16x16x16
            nn.ConvTranspose2d(in_channels = 16 , out_channels = 1, kernel_size=3, padding=1), 
            #nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'),                     # Upsampling to 32x32x1
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, h):
        DE = self.DE(h)
        DE = DE.reshape(-1,1,4,4)
        r = self.sigmoid(self.Decoder(DE))
        # return self.Decoder(h) 
        return DE,r

# %%  Autoencoder
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, x):
        mu,Std,LaVar = self.encoder(x)
        DE,r = self.decoder(LaVar)
        return mu,Std,DE,r
    
