# %% imports
import torch
import torch.nn as nn
# Build an Encoder for classfication
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
            # nn.MaxPool2d(2),                      # Downsampling to 2x2x16
            nn.Flatten(),                   # Flatten 2x2x16 to 1-dim
            nn.Linear(in_features=64*2*2,out_features=10),
            nn.Softmax(dim=1),
            # nn.Conv2d(in_channels = 16, out_channels = 1, kernel_size=3, padding=1), 
            #nn.ReLU(inplace=True),
            # nn.Flatten(),
            # nn.ReLU(inplace=True)
            # nn.MaxPool2d((2,1)),                      # Output to 1x2x1
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
            nn.Linear(in_features=10,out_features=2),
            # nn.Upsample(scale_factor=(2,1), mode='bilinear'),  
            # nn.Flatten(),
            # nn.ConvTranspose2d(1 , out_channels = 16, kernel_size=3, padding=1), # Padding = 1 ensures that the final output is the same size as input 
            nn.ReLU(inplace=True),
            # nn.Unflatten(1,(10,20,20)),
            nn.Upsample(scale_factor=(10,1), mode='bilinear'),                     # Upsampling to 2x2x16
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
class AE4(nn.Module):
    def __init__(self):
        super(AE4, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, x):
        h = self.encoder(x)
        r = self.decoder(h)
        return r, h
    

# %%
