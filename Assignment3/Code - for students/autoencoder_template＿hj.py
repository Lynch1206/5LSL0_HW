# %% imports
import torch
import torch.nn as nn

# %%  Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.Encoder = nn.Sequential(
            nn.Conv2d(1, out_channels = 16, kernel_size=3, padding=1), # Padding = 1 ensures that the final output is the same size as input 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                      # Downsampling to 16x16x16
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                      # Downsampling to 8x8x16
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                      # Downsampling to 4x4x16
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                      # Downsampling to 2x2x16
            nn.Conv2d(in_channels = 16, out_channels = 1, kernel_size=3, padding=1), 
            #nn.ReLU(inplace=True),
            nn.MaxPool2d((2,1)),                      # Output to 1x2x1
        )
        
    def forward(self, x):
        # use the created layers here
        return self.Encoder(x)   
    
# %%  Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # create layers here
        self.Decoder = nn.Sequential(
            nn.ConvTranspose2d(1 , out_channels = 16, kernel_size=3, padding=1), # Padding = 1 ensures that the final output is the same size as input 
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
        
    def forward(self, h):
        return self.Decoder(h) 
    
# %%  Autoencoder
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, x):
        h = self.encoder(x)
        r = self.decoder(h)
        return r, h
    

# %%  Classifier
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.CNN = nn.Sequential(
            nn.Conv2d(1, out_channels = 16, kernel_size=3, padding=1), # Padding = 1 ensures that the final output is the same size as input 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                      # Downsampling to 16x16x16
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                      # Downsampling to 8x8x16
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                      # Downsampling to 4x4x16
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                      # Downsampling to 2x2x16
            nn.Linear(16, 10) # Here we changed embedding dimensionality into 10d
            nn.Conv2d(in_channels = 16, out_channels = 1, kernel_size=3, padding=1), 
            #nn.ReLU(inplace=True),
            nn.MaxPool2d((2,1)),                      # Output to 1x2x1
        )
        
    def forward(self, x):
        # use the created layers here
        return self.Encoder(x)   